"""Adapt native NAOMi-derived optics without changing the RANA core."""
import numpy as np
from collections.abc import Mapping
from ..fields import FullFieldPSF
from ..config import RanaAddon
from ..providers import resolve_field
from ..operator import RanaOperator

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
        if isinstance(addon.field,Mapping) and set(addon.field) != set(optics['channels']):
            raise ValueError('Supply one RANA field per optical channel')
        from dataclasses import replace
        channels = {name: attach_rana(value,replace(addon,field=addon.field[name] if isinstance(addon.field,Mapping) else addon.field),
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
    else:
        field = resolve_field(field)
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
