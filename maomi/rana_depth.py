"""Depth-dependent complete PSF banks; positive linear depth interpolation."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .enhanced import FullFieldPSF, CubicFullFieldRenderer


@dataclass(frozen=True)
class DepthPSFBank:
    depths_um: np.ndarray
    fields: tuple

    def __post_init__(self):
        z = np.array(self.depths_um, dtype=float, copy=True)
        fields = tuple(self.fields)
        if z.ndim != 1 or len(z) < 2 or (np.diff(z) <= 0).any() or not np.isfinite(z).all() or (z < 0).any():
            raise ValueError('A depth bank requires >= 2 increasing nonnegative finite depths')
        if len(fields) != len(z) or any(not isinstance(f, FullFieldPSF) or f.kernels.ndim != 5 for f in fields):
            raise ValueError('One complete XYZ-kernel FullFieldPSF per depth is required')
        first = fields[0]
        for f in fields[1:]:
            if (f.kernels.shape != first.kernels.shape or f.voxel_size_um != first.voxel_size_um
                    or f.operator != first.operator or f.wavelength_nm != first.wavelength_nm
                    or not np.array_equal(f.x_um, first.x_um) or not np.array_equal(f.y_um, first.y_um)):
                raise ValueError('Depth fields must share shape, XY nodes, pitch, operator and wavelength')
        z.setflags(write=False)
        object.__setattr__(self, 'depths_um', z)
        object.__setattr__(self, 'fields', fields)

    @property
    def voxel_size_um(self):
        return self.fields[0].voxel_size_um

    @property
    def operator(self):
        return self.fields[0].operator

    @property
    def wavelength_nm(self):
        return self.fields[0].wavelength_nm

    @classmethod
    def from_provider(cls, provider, x_um, y_um, depths_um, *, voxel_size_um, wavelength_nm, operator='excitation', provenance=None):
        return cls(depths_um, tuple(FullFieldPSF.from_provider(
            lambda x, y, z=z: provider(x, y, float(z)), x_um, y_um,
            voxel_size_um=voxel_size_um, wavelength_nm=wavelength_nm, operator=operator,
            provenance={**(provenance or {}), 'focal_depth_um': float(z)}) for z in depths_um))

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        f = self.fields[0]
        np.savez_compressed(p, schema='maomi.depth_psf_bank.v1', depths_um=self.depths_um,
                            kernels=np.stack([x.kernels for x in self.fields]), x_um=f.x_um, y_um=f.y_um,
                            voxel_size_um=f.voxel_size_um, wavelength_nm=f.wavelength_nm, operator=f.operator,
                            provenance=json.dumps([x.provenance for x in self.fields], allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if str(d['schema']) != 'maomi.depth_psf_bank.v1':
                raise ValueError('Unsupported depth-bank schema')
            meta = json.loads(str(d['provenance']))
            if len(meta) != len(d['depths_um']) or len(d['kernels']) != len(d['depths_um']):
                raise ValueError('Malformed depth bank: inconsistent depth, kernel or provenance counts')
            return cls(d['depths_um'], tuple(FullFieldPSF(k, d['x_um'], d['y_um'], tuple(d['voxel_size_um']),
                                                       float(d['wavelength_nm']), str(d['operator']), m)
                                               for k, m in zip(d['kernels'], meta)))


class DepthFullFieldRenderer:
    """Cubic XY fields + linear focal-depth weights; no shared-kernel fallback.

    Source depth weights act before convolution for detection; scan depth
    weights act after correlation for excitation. Each bank is streamed once.
    """
    def __init__(self, bank, shape, *, voxel_size_um, origin_um=(0., 0.), z_origin_um=0., device='cpu'):
        if not isinstance(bank, DepthPSFBank) or len(shape) != 3:
            raise TypeError('Depth rendering requires DepthPSFBank and an XYZ volume')
        if not np.isfinite(z_origin_um):
            raise ValueError('z_origin_um must be finite')
        end = z_origin_um + shape[2]*voxel_size_um[2]
        if z_origin_um < bank.depths_um[0] or end > bank.depths_um[-1]:
            raise ValueError('Depth bank must cover the entire Z extent including edges')
        self.bank, self.shape = bank, tuple(shape)
        z = z_origin_um+(np.arange(shape[2])+.5)*voxel_size_um[2]
        self.weights = np.stack([np.interp(z, bank.depths_um, np.eye(len(bank.fields))[i]) for i in range(len(bank.fields))])
        self.renderers = [CubicFullFieldRenderer(f, shape, voxel_size_um=voxel_size_um, origin_um=origin_um, device=device)
                          for f in bank.fields]
        self.validation = {'depth_nodes_um': bank.depths_um.tolist(), 'checked_xyz_positions': int(np.prod(shape)),
                           'shared_psf_fallback': False, 'depth_interpolation': 'positive linear',
                           'fields': [r.validation for r in self.renderers]}

    def __call__(self, density):
        result = np.zeros(self.shape)
        for w, renderer in zip(self.weights, self.renderers):
            if not np.any(w):
                continue
            if self.bank.operator == 'detection':
                result += renderer(np.asarray(density)*w[None, None, :])
            else:
                result += renderer(density)*w[None, None, :]
        return result
