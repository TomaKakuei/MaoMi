"""RANA local-field operator; independent of native tissue and propagation."""
from collections import OrderedDict
import numpy as np
from .fields import FullFieldPSF
from .depth import DepthPSFBank
from .providers import resolve_field
from ._backends import get_backend


def _axis_weights(nodes, positions):
    ids = np.clip(np.searchsorted(nodes, positions, side='right')-1, 0, len(nodes)-2)
    t = np.clip((positions-nodes[ids])/(nodes[ids+1]-nodes[ids]), 0, 1)
    t = t*t*(3-2*t)
    weights = np.zeros((len(nodes), len(positions)))
    weights[ids, np.arange(len(positions))] = 1-t
    weights[ids+1, np.arange(len(positions))] = t
    return weights


class RanaOperator:
    def __init__(self, field, *, native=False, origin_um=(0., 0.), z_origin_um=0., device='cpu'):
        field = resolve_field(field)
        self.bank = field if isinstance(field, DepthPSFBank) else None
        self.fields = field.fields if self.bank is not None else (field,)
        if any(not isinstance(f, FullFieldPSF) or f.kernels.ndim != 5 for f in self.fields):
            raise TypeError('The acquisition add-on requires complete XYZ-kernel fields')
        self.field = self.fields[0]
        self.kernel_shape = self.field.kernels.shape[2:]
        self.voxel_size_um = self.field.voxel_size_um
        self.origin_um, self.z_origin_um = np.asarray(origin_um, float), float(z_origin_um)
        self.native, self.device = native, str(device)
        if (self.origin_um.shape != (2,) or not np.isfinite(self.origin_um).all()
                or not np.isfinite(self.z_origin_um)):
            raise ValueError('Supply finite XY and Z origins')
        self._backend = get_backend(self.device)
        self._cache = {}
        self._plans = OrderedDict()
        self._scan_kernels = {}
        self.settings = {'enabled': True, 'operator': 'native_scan' if native else self.field.operator,
            'interpolation': 'positive local cubic tensor weights',
            'field_nodes': list(self.field.kernels.shape[:2]), 'kernel_shape': list(self.kernel_shape),
            'depth_nodes_um': None if self.bank is None else self.bank.depths_um.tolist(),
            'source_masks': 'existing excitation and collection masks retained',
            'native_gain': 'source excitation mask' if native else 'custom kernel gain plus source masks',
            'tail_resampling': 'nonnegative intensity projection, integrated gain preserved',
            'shared_psf_fallback': False, 'device': self.device,
            'provenance': [f.provenance for f in self.fields]}

    def _weights(self, shape):
        shape = tuple(shape[:2])
        if shape not in self._cache:
            pitch = np.array(self.voxel_size_um[:2])
            lower = np.array([self.field.x_um[0], self.field.y_um[0]])
            upper = np.array([self.field.x_um[-1], self.field.y_um[-1]])
            if (self.origin_um < lower-1e-10).any() or (self.origin_um+np.array(shape)*pitch > upper+1e-10).any():
                raise ValueError('RANA field must cover the entire acquisition FOV')
            coords = [o+(np.arange(n)+.5)*p for o,n,p in zip(self.origin_um, shape, pitch)]
            self._cache[shape] = tuple(_axis_weights(nodes, q) for nodes,q in zip((self.field.x_um,self.field.y_um),coords))
        return self._cache[shape]

    def _depth_weights(self, z):
        if self.bank is None:
            return np.ones((1, np.size(z)))
        z = np.atleast_1d(z)
        if (z < self.bank.depths_um[0]-1e-10).any() or (z > self.bank.depths_um[-1]+1e-10).any():
            raise ValueError('RANA depth bank does not cover the requested acquisition depths')
        return np.stack([np.interp(z, self.bank.depths_um, row) for row in np.eye(len(self.fields))])

    def clear_cache(self):
        """Release prepared kernel FFTs and device-side interpolation weights."""
        self._plans.clear()
        self._scan_kernels.clear()
        self._cache.clear()

    @property
    def cache_info(self):
        names = ('kernel_fft_builds','cache_hits')
        return dict(plans=len(self._plans),
                    fft_bytes=sum(p.cache_bytes for p in self._plans.values()),
                    **{k:sum(p.stats[k] for p in self._plans.values()) for k in names})

    def _project(self, source, kernels, *, plane=False):
        from ._fft import LocalFFTPlan
        key = (tuple(source.shape),id(kernels),plane)
        if key not in self._plans:
            limit = max(2,2*len(self.fields))
            while len(self._plans) >= limit:
                self._plans.popitem(last=False)[1].clear()
            self._plans[key] = LocalFFTPlan(source.shape,kernels,self._weights(source.shape),
                plane=plane,detection=not self.native and self.field.operator == 'detection',
                correlate=not self.native and self.field.operator == 'excitation',
                device=self.device,cache_mb=256/limit)
        self._plans.move_to_end(key)
        return self._plans[key](source)

    def _device_array(self, a):
        return self._backend.array(a)

    def _finish(self, result):
        return np.maximum(self._backend.to_numpy(result),0.)

    @staticmethod
    def physical_tail(values):
        """Remove negative intensity ringing without changing integrated gain."""
        a = np.asarray(values)
        if np.iscomplexobj(a) or not np.isfinite(a).all() or a.sum(dtype=float) <= 0:
            raise ValueError('Tail intensity must have positive finite integrated gain')
        if np.any(a < 0):
            positive = np.maximum(a, 0)
            return positive*(a.sum(dtype=float)/positive.sum(dtype=float))
        return a

    @staticmethod
    def _density(volume):
        if np.iscomplexobj(volume):
            raise ValueError('Fluorescence must be real')
        a = np.asarray(volume, dtype=float)
        if a.ndim != 3 or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError('Supply finite nonnegative XYZ fluorescence')
        return a

    def static(self, volume):
        a = self._density(volume)
        z = self.z_origin_um+(np.arange(a.shape[2])+.5)*self.voxel_size_um[2]
        weights = self._depth_weights(z)
        a = self._device_array(a)
        result = a*0
        detection = not self.native and self.field.operator == 'detection'
        for f,w in zip(self.fields,weights):
            if not np.any(w):
                continue
            wd = self._device_array(w)
            result += self._project(a*wd if detection else a, f.kernels)*(1 if detection else wd)
        return self._finish(result)

    def scan_slab(self, slab, *, z_start=0, scan_avg=1):
        a = self._density(slab)
        if a.shape[2] != self.kernel_shape[2]:
            raise ValueError('Focal slab depth must match the RANA kernel')
        if not isinstance(scan_avg,(int,np.integer)) or scan_avg < 1:
            raise ValueError('scan_avg must be a positive integer')
        center = self.kernel_shape[2]//2
        z = self.z_origin_um+(z_start+np.arange(a.shape[2])+.5)*self.voxel_size_um[2]
        detection = not self.native and self.field.operator == 'detection'
        weights = self._depth_weights(z if detection else [z[center]])
        if scan_avg > 1:
            starts = np.arange(0,a.shape[2],scan_avg)
        a = self._device_array(a)
        result = a.sum(axis=2)*0
        for index,(f,w) in enumerate(zip(self.fields,weights)):
            if not np.any(w):
                continue
            source = a*self._device_array(w) if detection else a
            key = (index,scan_avg,detection)
            if key not in self._scan_kernels:
                kernels = f.kernels
                if detection:
                    ids = 2*center-np.arange(a.shape[2])
                    kernels = np.take(kernels,np.minimum(ids,a.shape[2]-1),axis=-1).copy()
                    kernels[...,ids >= a.shape[2]] = 0
                if scan_avg > 1:
                    kernels = np.add.reduceat(kernels,starts,axis=-1)
                self._scan_kernels[key] = kernels
            if scan_avg > 1:
                source = self._backend.group_sum(source,starts,scan_avg)
            result += self._project(source,self._scan_kernels[key],plane=True)*(1 if detection else w[0])
        return self._finish(result)
