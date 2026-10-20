"""Optional local-PSF operator for the existing static and movie pipelines.

Native kernels retain MATLAB offset conventions. Positive cubic tensor weights
interpolate complete local kernels; compact XY support permits tiled FFTs.
Native source-energy and collection masks, tails and detector stages stay in
their existing pipeline. Custom fields keep their explicit operator and gains.
"""
from dataclasses import dataclass
import numpy as np
from scipy import fft
from .enhanced import FullFieldPSF
from .rana_depth import DepthPSFBank
from .rana import RanaConfig


def _axis_weights(nodes, positions):
    ids = np.clip(np.searchsorted(nodes, positions, side='right')-1, 0, len(nodes)-2)
    t = np.clip((positions-nodes[ids])/(nodes[ids+1]-nodes[ids]), 0, 1)
    t = t*t*(3-2*t)
    weights = np.zeros((len(nodes), len(positions)))
    weights[ids, np.arange(len(positions))] = 1-t
    weights[ids+1, np.arange(len(positions))] = t
    return weights


@dataclass(frozen=True)
class RanaAddon:
    """field=None preserves all native per-position PSFs automatically.

    field may also be a FullFieldPSF, DepthPSFBank or RanaConfig: this explicitly
    supplies the focal optical model while preserving the acquisition stages.
    Custom kernels are in the scan pipeline's response units; native tissue
    masks are separate source weights. No implicit gain or exposure fitting.
    """
    field: object = None
    voxel_size_um: tuple | None = None
    origin_um: tuple = (0., 0.)
    z_origin_um: float = 0.
    device: str | None = None
    status_window: bool = True


class RanaOperator:
    def __init__(self, field, *, native=False, origin_um=(0., 0.), z_origin_um=0., device='cpu'):
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
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('device must be cpu or cuda[:index]')
        self._cache = {}
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

    def _convolve(self, source, kernel, *, plane=False, correlate=False):
        if correlate:
            kernel = np.flip(kernel, axis=(0, 1) if plane else (0, 1, 2))
        dims = (0, 1) if plane else (0, 1, 2)
        lengths = tuple(fft.next_fast_len(source.shape[d]+kernel.shape[d]-1) for d in dims)
        if self.device == 'cpu':
            spec = fft.rfftn(source, s=lengths, axes=dims)*fft.rfftn(kernel, s=lengths, axes=dims)
            if plane:
                spec = spec.sum(axis=2)
            full = fft.irfftn(spec, s=lengths, axes=dims)
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA-enabled PyTorch is required')
            a, k = [torch.as_tensor(np.array(v, order='C', copy=True), dtype=torch.float64, device=self.device) for v in (source,kernel)]
            spec = torch.fft.rfftn(a, s=lengths, dim=dims)*torch.fft.rfftn(k, s=lengths, dim=dims)
            if plane:
                spec = spec.sum(dim=2)
            full = torch.fft.irfftn(spec, s=lengths, dim=dims).cpu().numpy()
        centers = [(kernel.shape[d]-1-kernel.shape[d]//2) if correlate else kernel.shape[d]//2 for d in dims]
        return full[tuple(slice(c,c+source.shape[d]) for c,d in zip(centers,dims))]

    def _project(self, source, kernels, *, plane=False):
        wx, wy = self._weights(source.shape)
        result = np.zeros(source.shape[:2] if plane else source.shape)
        detection = not self.native and self.field.operator == 'detection'
        correlate = not self.native and self.field.operator == 'excitation'
        for i in range(len(wx)):
            for j in range(len(wy)):
                xi, yi = np.flatnonzero(wx[i]), np.flatnonzero(wy[j])
                if len(xi) == 0 or len(yi) == 0:
                    continue
                kernel = kernels[i,j]
                if detection:
                    # Source-local fields: weights act before convolution.
                    w = wx[i,:,None]*wy[j,None,:]
                    result += self._convolve(source*w[...,None], kernel, plane=plane)
                else:
                    # Scan-local fields: compute only their compact output tile,
                    # with enough input halo for an exact linear convolution.
                    lo = np.array([xi[0],yi[0]])
                    hi = np.array([xi[-1]+1,yi[-1]+1])
                    halo = np.array(kernel.shape[:2])
                    start = np.maximum(0, lo-halo)
                    end = np.minimum(source.shape[:2], hi+halo)
                    tile = source[start[0]:end[0],start[1]:end[1]]
                    response = self._convolve(tile, kernel, plane=plane, correlate=correlate)
                    response = response[lo[0]-start[0]:hi[0]-start[0],lo[1]-start[1]:hi[1]-start[1]]
                    w = wx[i,lo[0]:hi[0],None]*wy[j,None,lo[1]:hi[1]]
                    result[lo[0]:hi[0],lo[1]:hi[1]] += response*(w if plane else w[...,None])
        return result

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
        result = np.zeros_like(a)
        detection = not self.native and self.field.operator == 'detection'
        for f,w in zip(self.fields,weights):
            if not np.any(w):
                continue
            result += self._project(a*w if detection else a, f.kernels)*(1 if detection else w)
        return np.maximum(result,0.)

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
        result = np.zeros(a.shape[:2])
        for f,w in zip(self.fields,weights):
            if not np.any(w):
                continue
            kernels = f.kernels
            source = a*w if detection else a
            if detection:
                # At output focal plane center, offset is center - source_z.
                ids = 2*center-np.arange(a.shape[2])
                kernels = np.take(kernels, np.minimum(ids,a.shape[2]-1), axis=-1).copy()
                kernels[...,ids >= a.shape[2]] = 0
            if scan_avg > 1:
                starts = np.arange(0,a.shape[2],scan_avg)
                source = np.add.reduceat(source, starts, axis=2)
                kernels = np.add.reduceat(kernels, starts, axis=-1)
            result += self._project(source,kernels,plane=True)*(1 if detection else w[0])
        return np.maximum(result,0.)


def attach_rana(optics, rana=True, *, config=None, volume=None, device=None):
    """Return the original optics mapping with an optional acquisition operator.

    Existing callers keep scan_volume/iter_scan_volume and movie writers. False
    is an identity operation. Native scattering/pupils/beam inputs are computed
    before attachment; per-position PSFs are reused without another propagation.
    """
    if rana is False or rana is None:
        return optics
    addon = RanaAddon() if rana is True else rana if isinstance(rana,RanaAddon) else RanaAddon(field=rana)
    selected_device = addon.device or device or 'cpu'
    if 'channels' in optics:
        if isinstance(addon.field,dict) and set(addon.field) != set(optics['channels']):
            raise ValueError('Supply one RANA field per optical channel')
        from dataclasses import replace
        channels = {name: attach_rana(value,replace(addon,field=addon.field[name] if isinstance(addon.field,dict) else addon.field),
                    config=config,volume=volume,device=selected_device) for name,value in optics['channels'].items()}
        out = dict(optics, channels=channels)
        if volume is not None:
            out['image'] = sum(optics['beam_weights'][name]*value['image'] for name,value in channels.items())
        out['settings'] = dict(optics.get('settings',{}), rana={'enabled':True,'channels':list(channels)})
        return out
    field = addon.field
    native = field is None
    if native:
        pitch = addon.voxel_size_um if addon.voxel_size_um is not None else ((1/config.vres,)*3 if config is not None else None)
        if pitch is None:
            raise ValueError('Specify voxel_size_um when attaching native PSFs without OpticalConfig')
        points = np.asarray(optics['points_um'],float)
        x,y = np.unique(points[:,0]), np.unique(points[:,1])
        kernels = np.abs(np.asarray(optics['psf_3d'],float))
        if len(points) != len(x)*len(y) or len(np.unique(points,axis=0)) != len(points):
            raise ValueError('Native RANA requires a complete rectangular propagation grid')
        order = np.lexsort((points[:,1],points[:,0]))
        kernels = kernels[order]
        energy = kernels.sum(axis=(1,2,3))
        if (energy <= 0).any():
            raise ValueError('Every native local PSF must have positive energy')
        # The existing source mask already carries native local excitation gain.
        # Retain that gain there, and replace only the shared shape by local shapes.
        kernels *= float(np.asarray(optics['psf']).sum())/energy[:,None,None,None]
        field = FullFieldPSF(kernels.reshape(len(x),len(y),*kernels.shape[1:]),x,y,pitch,
            config.wavelength_um*1000 if config is not None else 920.,
            'excitation', {'source':'native per-position propagation','offset_convention':'MATLAB',
                           'local_gain':'retained in existing source mask'})
    elif isinstance(field,RanaConfig):
        field = field.build_field()
    expected_pitch = addon.voxel_size_um if addon.voxel_size_um is not None else ((1/config.vres,)*3 if config is not None else None)
    if expected_pitch is not None and not np.allclose(field.voxel_size_um, expected_pitch, rtol=1e-10, atol=1e-12):
        raise ValueError('RANA PSF voxel size must match the acquisition volume')
    operator = RanaOperator(field,native=native,origin_um=addon.origin_um,z_origin_um=addon.z_origin_um,device=selected_device)
    out = dict(optics, rana=operator)
    out['settings'] = dict(optics.get('settings',{}), rana=operator.settings)
    if volume is not None:
        a = np.asarray(volume,float)
        for key in ('mask','colmask'):
            if optics.get(key) is not None:
                a = a*np.asarray(optics[key])[...,None]
        out['image'] = operator.static(a)
    return out
