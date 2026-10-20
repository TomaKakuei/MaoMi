"""RayOptics scalar pupil -> XYZ angular-spectrum PSFs or native tissue pupil."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .rayoptics import rayoptics_psf
from .volume import PSFVolume, volume_psf_field


def _trace(model,field_index,num_rays):
    from rayoptics.raytr.analyses import RayGrid
    model.update_model()
    grid = RayGrid(model,f=field_index,num_rays=num_rays,foc=0.,clip_rays=True,value_if_none=np.nan)
    x,y,opd = np.asarray(grid.grid,float)
    return grid,x[:,0],y[0,:],opd


def rayoptics_psf_volume(model,*,voxel_size_um,shape,field_index=0,num_rays=33,
                        fft_size=128,magnification=None,specimen_index=1.33):
    """Propagate the traced focal complex field along specimen Z.

    Uses the same scalar/paraxial pupil mapping as rayoptics_psf. Angular-spectrum
    propagation is in a homogeneous specimen with explicit refractive index.
    +Z follows exp(+i*kz*z); resampling is at physical voxel centers, without
    slice normalization. Tissue scattering is composed via the pupil adapter.
    """
    pitch = tuple(float(p) for p in voxel_size_um)
    shape = tuple(int(n) for n in shape)
    if len(shape)!=3 or min(shape)<1 or len(pitch)!=3 or min(pitch)<=0 or specimen_index<=0:
        raise ValueError("Positive XYZ shape, pitch and specimen refractive index required")
    base = rayoptics_psf(model,num_rays=num_rays,fft_size=fft_size,
                        field_index=field_index,magnification=magnification)
    grid,_,_,opd = _trace(model,field_index,num_rays)
    valid = np.isfinite(opd)
    aperture = np.where(valid,np.exp(2j*np.pi*np.nan_to_num(opd)),0.)
    padded = np.zeros((fft_size,fft_size),complex)
    start = [fft_size//2-n//2 for n in opd.shape]
    padded[start[0]:start[0]+opd.shape[0],start[1]:start[1]+opd.shape[1]] = aperture
    focus = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))/(fft_size*num_rays)
    spectral = np.fft.fft2(focus)
    dx,dy = base.pixel_size_um
    fx,fy = np.meshgrid(np.fft.fftfreq(fft_size,dx),np.fft.fftfreq(fft_size,dy),indexing="ij")
    k2 = (specimen_index/(base.wavelength_nm/1000))**2-fx*fx-fy*fy
    kz = 2*np.pi*np.sqrt(np.maximum(k2,0))
    spectral[k2<0] = 0
    axes = [(np.arange(fft_size)-fft_size//2)*d for d in (dx,dy)]
    q = np.stack(np.meshgrid(*[(np.arange(n)-n//2)*d for n,d in zip(shape[:2],pitch)],indexing="ij"),-1)
    slices = []
    for z in (np.arange(shape[2])-shape[2]//2)*pitch[2]:
        intensity = np.abs(np.fft.ifft2(spectral*np.exp(1j*kz*z)))**2/(dx*dy)
        slices.append(RegularGridInterpolator(axes,intensity,bounds_error=False,fill_value=0)(q))
    return PSFVolume(np.stack(slices,axis=-1),pitch,base.wavelength_nm,"intensity",
        dict(provider="RayOptics + scalar angular spectrum",field_index=field_index,
             specimen_index=specimen_index,source=base.provenance,axial_origin="shape//2",
             normalization="common incident pupil amplitude; no slice normalization"))


def rayoptics_volume_field(model,x_um,y_um,field_indices,*,voxel_size_um,shape,
                           photon_order=2,normalization="reference_sum",**options):
    """Complete XYZ effective excitation field, directly accepted by RanaAddon."""
    indices = np.asarray(field_indices)
    if indices.shape!=(len(x_um),len(y_um)) or (indices<0).any() or not np.equal(indices,np.floor(indices)).all():
        raise ValueError("field_indices must match the complete XY grid")
    lookup = {(float(x),float(y)):int(indices[i,j]) for i,x in enumerate(x_um) for j,y in enumerate(y_um)}
    cache = {}
    def provider(x,y):
        i = lookup[x,y]
        if i not in cache:
            cache[i] = rayoptics_psf_volume(model,voxel_size_um=voxel_size_um,shape=shape,field_index=i,**options)
        return cache[i]
    return volume_psf_field(provider,x_um,y_um,photon_order=photon_order,normalization=normalization)


def rayoptics_native_pupil(model,config,*,field_index=0,num_rays=33):
    """Map traced normalized pupil amplitude/OPD onto MaoMi's objective aperture.

    The caller asserts that the traced exit pupil represents this objective's
    normalized pupil. Matching excitation wavelengths are required. This complex
    pupil enters native A/B/C tissue propagation, composing aberration BEFORE
    scattering and multiphoton excitation, rather than convolving two PSFs.
    """
    import torch
    from ..optics import pupil,legacy
    grid,ux,uy,opd = _trace(model,field_index,num_rays)
    if not np.isclose(float(grid.wvl),config.wavelength_um*1000,rtol=1e-6):
        raise ValueError("RayOptics and native excitation wavelengths must match")
    if (np.diff(ux)<=0).any() or (np.diff(uy)<=0).any():
        raise ValueError("Increasing rectilinear pupil coordinates required")
    base = pupil(config,device="cpu").numpy()
    v,p = config.parameters()
    fl = config.focal_length_mm/1000
    spacing = max(legacy.gaussian_beam_size(p,fl*1e6))/1e6/min(base.shape)
    radius = np.tan(np.arcsin(config.objNA/config.n))*fl
    # Native pupil arrays are indexed by the first/second transverse coordinates.
    q = np.stack(np.meshgrid(*[(np.arange(n)-n/2)*spacing/radius for n in base.shape],indexing="ij"),-1)
    valid = np.isfinite(opd)
    phase = RegularGridInterpolator((ux,uy),np.nan_to_num(opd),bounds_error=False,fill_value=0)(q)
    transmission = RegularGridInterpolator((ux,uy),valid.astype(float),method="nearest",bounds_error=False,fill_value=0)(q)
    return torch.as_tensor(base*transmission*np.exp(2j*np.pi*phase),dtype=torch.complex64)
