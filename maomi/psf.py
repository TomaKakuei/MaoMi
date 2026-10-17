"""Public PSF data contracts, independent of optical providers and rendering.

Coordinates use specimen-plane micrometres, arrays use XY order, and kernel
origins are shape // 2. Intensity kernels describe redistribution, not throughput.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from typing import Protocol


@dataclass(frozen=True)
class SampledPSF:
    """Unit-sum redistribution kernel, centered at shape // 2.

    Pixel pitch is in specimen-plane micrometres. Throughput and illumination
    belong to the acquisition model, not this normalized kernel. This contract
    describes a shift-invariant, incoherent, monochromatic 2-D detection PSF.
    """
    intensity: np.ndarray
    pixel_size_um: tuple[float, float]
    wavelength_nm: float
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        a = np.array(self.intensity, dtype=np.float64, copy=True)
        pitch = tuple(float(x) for x in self.pixel_size_um)
        if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("PSF must be a finite, nonnegative 2-D intensity array")
        if not np.isclose(a.sum(), 1., rtol=1e-10, atol=1e-12):
            raise ValueError("PSF must have unit sum; use SampledPSF.normalized explicitly")
        if len(pitch) != 2 or not np.isfinite(pitch).all() or min(pitch) <= 0:
            raise ValueError("Specify two positive specimen-plane pixel pitches in micrometres")
        if not np.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive and finite")
        meta = json.loads(json.dumps(self.provenance, allow_nan=False))
        a.setflags(write=False)
        object.__setattr__(self, "intensity", a)
        object.__setattr__(self, "pixel_size_um", pitch)
        object.__setattr__(self, "provenance", meta)

    @classmethod
    def normalized(cls, intensity, pixel_size_um, wavelength_nm, provenance=None):
        a = np.asarray(intensity, dtype=np.float64)
        if not np.isfinite(a).all() or (a < 0).any() or not np.isfinite(a.sum()) or a.sum() <= 0:
            raise ValueError("Cannot normalize an invalid or zero-energy intensity PSF")
        return cls(a / a.sum(), pixel_size_um, wavelength_nm, provenance or {})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, intensity=self.intensity,
                            pixel_size_um=self.pixel_size_um, wavelength_nm=self.wavelength_nm,
                            axes="XY", normalization="unit_sum", provenance=json.dumps(self.provenance))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if str(d["axes"]) != "XY" or str(d["normalization"]) != "unit_sum":
                raise ValueError("Unsupported PSF axes or normalization")
            return cls(d["intensity"], tuple(d["pixel_size_um"]), float(d["wavelength_nm"]),
                       json.loads(str(d["provenance"])))


class PSFProvider(Protocol):
    """Callable returning a local kernel at specimen coordinates (x_um, y_um).

    A function or callable object is sufficient: no registration or inheritance.
    """
    def __call__(self, x_um: float, y_um: float) -> SampledPSF: ...


@dataclass(frozen=True)
class PSFField:
    """Kernels on a rectilinear source-position grid, each with unit sum.

    Axes: field_X, field_Y, kernel_X, kernel_Y. Bilinear interpolation is applied
    at emitting positions, with nearest edge extension outside the grid. A
    singleton grid axis is constant. This is an incoherent, monochromatic 2-D
    operator, not a complex-field model. Sampling is common to every kernel.
    """
    intensity: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    pixel_size_um: tuple[float, float]
    wavelength_nm: float
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        a = np.array(self.intensity, dtype=np.float64, copy=True)
        coords = [np.array(c, dtype=float, copy=True) for c in (self.x_um, self.y_um)]
        for c in coords:
            if c.ndim != 1 or c.size == 0 or not np.isfinite(c).all() or (np.diff(c) <= 0).any():
                raise ValueError("PSF field coordinates must be finite and strictly increasing")
        if a.ndim != 4 or a.shape[:2] != tuple(len(c) for c in coords):
            raise ValueError("PSF field axes must be field_X, field_Y, kernel_X, kernel_Y")
        for kernel in a.reshape((-1, *a.shape[-2:])):
            checked = SampledPSF(kernel, self.pixel_size_um, self.wavelength_nm, self.provenance)
        a.setflags(write=False)
        for c in coords:
            c.setflags(write=False)
        object.__setattr__(self, "intensity", a)
        object.__setattr__(self, "x_um", coords[0])
        object.__setattr__(self, "y_um", coords[1])
        object.__setattr__(self, "pixel_size_um", checked.pixel_size_um)
        object.__setattr__(self, "provenance", checked.provenance)

    @classmethod
    def from_provider(cls, provider: PSFProvider, x_um, y_um):
        """Sample an optical program once per node; never silently resample."""
        x, y = np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float)
        for c in (x, y):
            if c.ndim != 1 or c.size == 0 or not np.isfinite(c).all() or (np.diff(c) <= 0).any():
                raise ValueError("PSF field coordinates must be finite and strictly increasing")
        kernels, records, first = [], [], None
        for xx in x:
            row = []
            for yy in y:
                p = provider(float(xx), float(yy))
                if not isinstance(p, SampledPSF):
                    raise TypeError("A PSF provider must return SampledPSF")
                if first is None:
                    first = p
                if (p.intensity.shape != first.intensity.shape
                        or not np.allclose(p.pixel_size_um, first.pixel_size_um, rtol=1e-10, atol=1e-12)
                        or not np.isclose(p.wavelength_nm, first.wavelength_nm, rtol=1e-10, atol=1e-12)):
                    raise ValueError("Provider kernels must share shape, pitch and wavelength; resample explicitly")
                row.append(p.intensity)
                records.append(dict(x_um=float(xx), y_um=float(yy), provider=p.provenance))
            kernels.append(row)
        return cls(np.array(kernels), x, y, first.pixel_size_um, first.wavelength_nm,
                   {"samples": records, "interpolation": "bilinear at source; nearest edge extension"})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, intensity=self.intensity, x_um=self.x_um, y_um=self.y_um,
                            pixel_size_um=self.pixel_size_um, wavelength_nm=self.wavelength_nm,
                            axes="field_X,field_Y,kernel_X,kernel_Y", normalization="unit_sum",
                            interpolation="source_bilinear_edge", provenance=json.dumps(self.provenance))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if (str(d["axes"]) != "field_X,field_Y,kernel_X,kernel_Y"
                    or str(d["normalization"]) != "unit_sum"
                    or str(d["interpolation"]) != "source_bilinear_edge"):
                raise ValueError("Unsupported PSF field axes, normalization or interpolation")
            return cls(d["intensity"], d["x_um"], d["y_um"], tuple(d["pixel_size_um"]),
                       float(d["wavelength_nm"]), json.loads(str(d["provenance"])))


def resample_psf(psf, pixel_size_um, shape):
    """Explicit linear intensity resampling about the discrete kernel origin.

    Zero outside supplied support; renormalize the retained kernel. This is an
    approximation, not optical propagation. The mass estimate before normalization
    is recorded to expose support changes; throughput is specified separately.
    """
    from scipy.ndimage import map_coordinates
    pitch = np.asarray(pixel_size_um, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("Specify two positive specimen-plane pixel pitches")
    if len(shape) != 2 or any(int(n) != n or n < 2 for n in shape):
        raise ValueError("shape must contain two integer dimensions >= 2")
    shape = tuple(int(n) for n in shape)
    coords = [(np.arange(n)-n//2)*step/old + size//2
              for n, step, old, size in zip(shape, pitch, psf.pixel_size_um, psf.intensity.shape)]
    a = map_coordinates(psf.intensity, np.array(np.meshgrid(*coords, indexing="ij")),
                        order=1, mode="constant", cval=0., prefilter=False)
    mass = float(a.sum()*np.prod(pitch)/np.prod(psf.pixel_size_um))
    return SampledPSF.normalized(a, tuple(pitch), psf.wavelength_nm,
        {"source": psf.provenance, "resampling": "linear intensity; zero extension; unit-sum renormalization",
         "source_pixel_size_um": psf.pixel_size_um, "source_shape": list(psf.intensity.shape),
         "pre_normalization_mass_estimate": mass})


def pupil_to_intensity(opd_waves, fft_size=256, amplitude=None):
    """Scalar diffraction from OPD in waves; NaN marks blocked/untraced rays.

    A finite zero OPD is a valid ray. No sentinel is inferred from phase values.
    Odd and even pupils are embedded with their discrete origin at N // 2.
    """
    opd = np.asarray(opd_waves, dtype=float)
    if opd.ndim != 2 or min(opd.shape) < 2:
        raise ValueError("OPD must be a 2-D grid")
    if int(fft_size) != fft_size or fft_size < max(opd.shape):
        raise ValueError("fft_size must be an integer no smaller than the pupil grid")
    amp = np.ones_like(opd) if amplitude is None else np.asarray(amplitude, dtype=float)
    if amp.shape != opd.shape or not np.isfinite(amp).all() or (amp < 0).any():
        raise ValueError("Pupil amplitude must match the OPD and be finite and nonnegative")
    valid = np.isfinite(opd) & (amp > 0)
    if not valid.any():
        raise ValueError("No transmitting pupil samples")
    pupil = np.zeros(opd.shape, dtype=np.complex128)
    pupil[valid] = amp[valid] * np.exp(2j * np.pi * opd[valid])
    n = int(fft_size)
    padded = np.zeros((n, n), dtype=np.complex128)
    start = [n // 2 - s // 2 for s in opd.shape]
    padded[start[0]:start[0]+opd.shape[0], start[1]:start[1]+opd.shape[1]] = pupil
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))) ** 2
    return intensity / intensity.sum()
