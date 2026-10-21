"""Three-dimensional wave/intensity providers and effective multiphoton PSFs."""
from dataclasses import dataclass, field
import numpy as np
from ..rana.fields import FullFieldPSF


@dataclass(frozen=True)
class PSFVolume:
    """XYZ samples: intensity, complex electric field, or effective excitation.

    The center is shape//2; pitch is in specimen micrometres. Complex fields
    may be XYZ or XYZ-by-polarization; orthogonal field components add in |E|².
    No normalization or multiphoton power is inferred.
    """
    samples: np.ndarray
    voxel_size_um: tuple
    wavelength_nm: float
    kind: str = "intensity"
    provenance: dict = field(default_factory=dict)

    def effective(self,photon_order=2):
        a = np.asarray(self.samples)
        if self.kind not in ("intensity","complex_field","effective_excitation"):
            raise ValueError("Declare intensity, complex_field or effective_excitation")
        if photon_order not in (1,2,3):
            raise ValueError("photon_order must be 1, 2 or 3")
        if len(self.voxel_size_um)!=3 or not np.isfinite(self.voxel_size_um).all() or min(self.voxel_size_um)<=0:
            raise ValueError("Three positive specimen voxel pitches required")
        if not np.isfinite(self.wavelength_nm) or self.wavelength_nm<=0 or not np.isfinite(a).all():
            raise ValueError("Finite wavelength and samples required")
        if self.kind == "complex_field":
            a = np.abs(a)**2
            if a.ndim==4:
                a = a.sum(axis=-1)
        elif np.iscomplexobj(a):
            raise ValueError("Complex values require kind='complex_field'")
        if a.ndim!=3 or min(a.shape)<1 or (a<0).any() or a.sum()<=0:
            raise ValueError("Provider must supply a positive-energy XYZ kernel")
        out = np.asarray(a,float) if self.kind=="effective_excitation" else np.asarray(a,float)**photon_order
        if not np.isfinite(out).all():
            raise ValueError("Effective excitation overflow")
        return out


def volume_psf_field(provider,x_um,y_um,*,photon_order=2,normalization="none"):
    """Sample an XYZ provider, preserving local relative gain.

    normalization='reference_sum' uses ONE common divisor (the first node's
    effective-kernel sum). It never independently normalizes each local PSF.
    A depth-dependent provider can call this per depth and build DepthPSFBank.
    """
    if normalization not in ("none","reference_sum"):
        raise ValueError("normalization must be none or reference_sum")
    kernels,records,first = [],[],None
    for x in x_um:
        row = []
        for y in y_um:
            v = provider(float(x),float(y))
            if not isinstance(v,PSFVolume):
                raise TypeError("The provider must return PSFVolume")
            a = v.effective(photon_order)
            if first is None:
                first = (a.shape,tuple(v.voxel_size_um),v.wavelength_nm)
            if a.shape!=first[0] or not np.allclose(v.voxel_size_um,first[1],rtol=1e-12,atol=0) or v.wavelength_nm!=first[2]:
                raise ValueError("Provider volumes must share XYZ grid and wavelength")
            row.append(a)
            records.append(dict(x_um=x,y_um=y,kind=v.kind,source=v.provenance))
        kernels.append(row)
    a = np.asarray(kernels)
    divisor = float(a[0,0].sum()) if normalization=="reference_sum" else 1.
    return FullFieldPSF(a/divisor,x_um,y_um,first[1],first[2],"excitation",
                        dict(provider="XYZ wave/intensity adapter",photon_order=photon_order,
                             normalization=normalization,common_divisor=divisor,samples=records))
