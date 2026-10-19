"""RANA wave optics and acquisition: explicit spatial, temporal and spectral units.

Richards-Wolf entrance-angle quadrature, s/p interface transmission and a
paraxially refocused layered OPD. All amplitudes share one reference scale;
local kernels are never independently normalized. See docs/RANA_PHYSICS.md.
"""
from dataclasses import dataclass, field
from itertools import product
import numpy as np
from scipy.special import roots_legendre


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class StratifiedMedium:
    immersion_index: float = 1.33
    specimen_index: float = 1.36
    glass_index: float = 1.52
    design_glass_index: float = 1.52
    glass_um: float = 170.
    design_glass_um: float = 170.

    def __post_init__(self):
        for name in ('immersion_index', 'specimen_index', 'glass_index', 'design_glass_index'):
            _positive(getattr(self, name), name)
        for name in ('glass_um', 'design_glass_um'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

    def opd_um(self, q, depth_um):
        """Layered path difference with piston and paraxial defocus removed.

        q = n sin(theta); depth is measured below the specimen surface.
        The immersion displacement cancels the coefficient of q**2.
        """
        if not np.isfinite(depth_um) or depth_um < 0:
            raise ValueError('Focal depth must be finite and nonnegative')
        ni, ns, ng, nd = self.immersion_index, self.specimen_index, self.glass_index, self.design_glass_index
        di = -ni * (depth_um/ns + self.glass_um/ng - self.design_glass_um/nd)
        def path(q):
            return (depth_um*np.sqrt(ns*ns-q*q) + di*np.sqrt(ni*ni-q*q)
                    + self.glass_um*np.sqrt(ng*ng-q*q) - self.design_glass_um*np.sqrt(nd*nd-q*q))
        return path(q) - path(0.)


@dataclass(frozen=True)
class PulseSpectrum:
    """Gaussian transform-limited intensity FWHM and quadratic spectral phase.

    Coherent frequency components interfere BEFORE intensity is raised to the
    multiphoton order. Fluence is fixed as dispersion changes, not peak power.
    Frequencies are uniform in angular frequency (radians / femtosecond).
    """
    wavelength_nm: float = 920.
    fwhm_fs: float = 100.
    gdd_fs2: float = 0.
    samples: int = 33
    time_oversample: int = 4

    def __post_init__(self):
        for name in ('wavelength_nm', 'fwhm_fs'):
            _positive(getattr(self, name), name)
        if not np.isfinite(self.gdd_fs2):
            raise ValueError('gdd_fs2 must be finite')
        if isinstance(self.samples, bool) or int(self.samples) != self.samples or self.samples < 5 or self.samples % 2 != 1:
            raise ValueError('Use an odd spectral sample count >= 5')
        if isinstance(self.time_oversample, bool) or int(self.time_oversample) != self.time_oversample or self.time_oversample < 2:
            raise ValueError('time_oversample must be an integer >= 2')
        # Reject a wrapped pulse rather than hiding insufficient temporal coverage.
        if self.time_period_fs < 8*self.broadened_fwhm_fs:
            raise ValueError('Increase pulse.samples: temporal window must cover eight broadened FWHMs')
        if (self.angular_frequencies <= 0).any():
            raise ValueError('Pulse bandwidth extends to nonpositive optical frequencies')

    @property
    def detuning(self):
        return np.linspace(-11*np.sqrt(np.log(2))/self.fwhm_fs,
                           11*np.sqrt(np.log(2))/self.fwhm_fs, self.samples)

    @property
    def angular_frequencies(self):
        return 2*np.pi*299.792458/self.wavelength_nm + self.detuning

    @property
    def wavelengths_nm(self):
        return 2*np.pi*299.792458/self.angular_frequencies

    @property
    def broadened_fwhm_fs(self):
        return self.fwhm_fs*np.sqrt(1+(4*np.log(2)*self.gdd_fs2/self.fwhm_fs**2)**2)

    @property
    def time_period_fs(self):
        return 2*np.pi/(self.detuning[1]-self.detuning[0])

    def integrate(self, fields, order=2):
        """Return integral (sum_c |E_c(t)|**2)**order dt for unit input fluence.

        fields axes: frequency, ..., complex vector component (1 or 3).
        Result has units fs**(1-order) times the spatial response.
        """
        fields = np.asarray(fields, dtype=complex)
        if fields.shape[0] != self.samples or fields.shape[-1] not in (1, 3) or not np.isfinite(fields).all():
            raise ValueError('Expected finite fields with frequency first and vector components last')
        if isinstance(order, bool) or int(order) != order or order < 1:
            raise ValueError('Multiphoton order must be a positive integer')
        dw = self.detuning
        spectrum = np.exp(-self.fwhm_fs**2*dw**2/(8*np.log(2)) + .5j*self.gdd_fs2*dw**2)
        n = self.time_oversample*self.samples
        dt = self.time_period_fs/n
        # Parseval: sum_t |ifft(A)*n|**2 dt = T sum_frequency |A|**2.
        spectrum /= np.sqrt(self.time_period_fs*np.sum(np.abs(spectrum)**2))
        flat = fields.reshape(self.samples, -1, fields.shape[-1])
        out = np.empty(flat.shape[1])
        # Bound temporal scratch to <= ~16 MiB, even for large 3-D kernels.
        chunk = max(1, 1_000_000//(n*fields.shape[-1]))
        bins = np.arange(-(self.samples//2), self.samples//2+1) % n
        for start in range(0, flat.shape[1], chunk):
            part = flat[:, start:start+chunk]
            padded = np.zeros((n, part.shape[1], part.shape[2]), dtype=complex)
            padded[bins] = spectrum[:, None, None]*part
            temporal = np.fft.ifft(padded, axis=0)*n
            intensity = np.sum(np.abs(temporal)**2, axis=-1)
            out[start:start+chunk] = np.sum(intensity**order, axis=0)*dt
        return out.reshape(fields.shape[1:-1])


@dataclass(frozen=True)
class EmissionResponse:
    """Photon spectral density and collection/QE on an increasing nm grid."""
    wavelength_nm: tuple
    photon_density: tuple
    efficiency: tuple

    def collection_fraction(self):
        w, s, e = [np.asarray(a, dtype=float) for a in (self.wavelength_nm, self.photon_density, self.efficiency)]
        if (w.ndim != 1 or len(w) < 2 or s.shape != w.shape or e.shape != w.shape
                or not all(np.isfinite(a).all() for a in (w, s, e)) or (np.diff(w) <= 0).any()
                or (w <= 0).any() or (s < 0).any() or (e < 0).any() or (e > 1).any()):
            raise ValueError('Use increasing positive wavelengths, nonnegative photon density and efficiency in [0,1]')
        total = np.trapezoid(s, w)
        if total <= 0:
            raise ValueError('Emission spectrum must have positive integral')
        return float(np.trapezoid(s*e, w)/total)


class VectorialOptics:
    """Polar-quadrature focusing through plane interfaces, in relative field units.

    Jones polarization is normalized once at the entrance pupil. Custom pupil
    OPD/amplitude callables receive (q/NA, phi, wavelength_nm, position_xyz_um).
    They can carry field-dependent optics from ray tracing or measured pupils.
    """
    def __init__(self, *, na=.9, medium=None, polarization=(1., 0.), vectorial=True,
                 radial_samples=20, azimuth_samples=48, opd=None, amplitude=None,
                 reference_wavelength_nm=920.):
        self.medium = medium or StratifiedMedium()
        _positive(na, 'na')
        _positive(reference_wavelength_nm, 'reference_wavelength_nm')
        if na >= min(self.medium.immersion_index, self.medium.specimen_index,
                     self.medium.glass_index, self.medium.design_glass_index):
            raise ValueError('NA must be below every refractive index (propagating rays)')
        for n, minimum in ((radial_samples, 4), (azimuth_samples, 8)):
            if isinstance(n, bool) or int(n) != n or n < minimum:
                raise ValueError('Use integer radial_samples >= 4 and azimuth_samples >= 8')
        jones = np.asarray(polarization, dtype=complex)
        if jones.shape != (2,) or not np.isfinite(jones).all() or np.linalg.norm(jones) == 0:
            raise ValueError('Supply a nonzero finite two-component Jones polarization')
        self.na, self.vectorial = float(na), bool(vectorial)
        self.radial_samples, self.azimuth_samples = radial_samples, azimuth_samples
        self.polarization = jones/np.linalg.norm(jones)
        self.opd, self.amplitude = opd, amplitude
        self.reference_wavelength_nm = reference_wavelength_nm
        ni = self.medium.immersion_index
        t, w = roots_legendre(radial_samples)
        theta_max = np.arcsin(na/ni)
        theta, wt = (t+1)*theta_max/2, w*theta_max/2
        theta, phi = np.meshgrid(theta, np.arange(azimuth_samples)*2*np.pi/azimuth_samples, indexing='ij')
        self.phi, theta = phi.ravel(), theta.ravel()
        self.q = ni*np.sin(theta)
        self.qx, self.qy = self.q*np.cos(self.phi), self.q*np.sin(self.phi)
        self.qz = np.sqrt(self.medium.specimen_index**2-self.q**2)
        self.weights = np.repeat(wt, azimuth_samples)*np.sin(theta)*(2*np.pi/azimuth_samples)*np.sqrt(np.cos(theta))
        self.weights /= self.weights.sum()  # One common reference, not node-wise PSF normalization.
        qs, qp = np.ones_like(theta), np.ones_like(theta)
        for n1, n2 in ((ni, self.medium.glass_index), (self.medium.glass_index, self.medium.specimen_index)):
            c1, c2 = np.sqrt(1-(self.q/n1)**2), np.sqrt(1-(self.q/n2)**2)
            qs *= 2*n1*c1/(n1*c1+n2*c2)
            qp *= 2*n1*c1/(n2*c1+n1*c2)
        if self.vectorial:
            c, s = self.qz/self.medium.specimen_index, self.q/self.medium.specimen_index
            cp, sp = np.cos(self.phi), np.sin(self.phi)
            ex, ey = self.polarization
            self.vectors = np.stack(((qs*sp**2+qp*c*cp**2)*ex+(qp*c-qs)*sp*cp*ey,
                                     (qp*c-qs)*sp*cp*ex+(qs*cp**2+qp*c*sp**2)*ey,
                                     -qp*s*(cp*ex+sp*ey)), axis=-1)
        else:
            self.vectors = ((qs+qp)/2)[:, None].astype(complex)

    def electric_field(self, axes_um, *, wavelength_nm=920., position_um=(0., 0., 0.)):
        """Complex Ex/Ey/Ez on an offset XYZ mesh, including local focal depth."""
        _positive(wavelength_nm, 'wavelength_nm')
        axes = [np.asarray(a, dtype=float) for a in axes_um]
        position = np.asarray(position_um, dtype=float)
        if (len(axes) != 3 or any(a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() for a in axes)
                or position.shape != (3,) or not np.isfinite(position).all() or position[2] < 0):
            raise ValueError('Use three finite nonempty offset axes and a nonnegative focal depth')
        path = self.medium.opd_um(self.q, position[2])
        if self.opd is not None:
            path = path + self.opd(self.q/self.na, self.phi, wavelength_nm, position)
        amplitude = 1. if self.amplitude is None else np.asarray(self.amplitude(self.q/self.na, self.phi, wavelength_nm, position))
        if np.iscomplexobj(path) or np.iscomplexobj(amplitude) or not np.isfinite(amplitude).all() or (np.asarray(amplitude) < 0).any() or not np.isfinite(path).all():
            raise ValueError('Pupil OPD must be finite; amplitude must be real, finite and nonnegative')
        k = 2*np.pi/(wavelength_nm/1000)
        weight = (self.weights*amplitude*np.exp(1j*k*path)*self.reference_wavelength_nm/wavelength_nm)[:, None]*self.vectors
        ex = np.exp(1j*k*axes[0][:, None]*self.qx)
        ey = np.exp(1j*k*axes[1][:, None]*self.qy)
        ez = np.exp(1j*k*axes[2][:, None]*self.qz)
        out = np.empty((*map(len, axes), self.vectors.shape[-1]), dtype=complex)
        # Separable angular quadrature avoids a giant voxel-by-ray matrix.
        for z in range(len(axes[2])):
            for c in range(self.vectors.shape[-1]):
                out[:, :, z, c] = (ex*(weight[:, c]*ez[z])) @ ey.T
        return out

    def response(self, axes_um, position_um, *, wavelength_nm=920., order=2, pulse=None):
        if isinstance(order, bool) or int(order) != order or order < 1:
            raise ValueError('Multiphoton order must be a positive integer')
        if pulse is None:
            return np.sum(np.abs(self.electric_field(axes_um, wavelength_nm=wavelength_nm,
                                                     position_um=position_um))**2, axis=-1)**order
        fields = np.stack([self.electric_field(axes_um, wavelength_nm=wl, position_um=position_um)
                           for wl in pulse.wavelengths_nm])
        return pulse.integrate(fields, order=order)


@dataclass(frozen=True)
class Acquisition:
    """Piecewise-constant source density integrated over voxels and a scan path.

    scan_offsets_um / dwell_weights specify arbitrary discrete trajectory
    quadrature; weights are fractions of dwell_seconds. Output kernels contain
    voxel volume * dwell seconds * collection fraction, without renormalization.
    """
    voxel_order: int = 2
    scan_offsets_um: tuple = ((0., 0., 0.),)
    dwell_weights: tuple = (1.,)
    dwell_seconds: float = 1.

    def __post_init__(self):
        if isinstance(self.voxel_order, bool) or int(self.voxel_order) != self.voxel_order or self.voxel_order < 1:
            raise ValueError('voxel_order must be a positive integer')
        s, w = np.asarray(self.scan_offsets_um, dtype=float), np.asarray(self.dwell_weights, dtype=float)
        if (s.ndim != 2 or s.shape[1] != 3 or w.shape != (len(s),) or len(s) == 0
                or not np.isfinite(s).all() or not np.isfinite(w).all() or (w < 0).any()
                or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError('Supply finite XYZ scan offsets and nonnegative dwell fractions summing to one')
        _positive(self.dwell_seconds, 'dwell_seconds')

    @classmethod
    def line(cls, displacement_um, *, samples=3, voxel_order=2, dwell_seconds=1.):
        if isinstance(samples, bool) or int(samples) != samples or samples < 1:
            raise ValueError('samples must be a positive integer')
        delta = np.asarray(displacement_um, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError('Supply finite XYZ scan displacement')
        q, w = roots_legendre(samples)
        return cls(voxel_order, tuple(map(tuple, q[:, None]*delta/2)), tuple(w/2), dwell_seconds)

    def kernel(self, optics, shape, pitch_um, position_um, *, wavelength_nm=920., order=2, pulse=None, emission=None):
        """Sample the optical response at quadrature points, before discretization."""
        pitch = np.asarray(pitch_um, dtype=float)
        if (len(shape) != 3 or any(isinstance(n, bool) or int(n) != n or n < 1 for n in shape)
                or pitch.shape != (3,) or not np.isfinite(pitch).all() or (pitch <= 0).any()):
            raise ValueError('Use three positive integer kernel sizes and three positive pitches')
        axes = [(np.arange(n)-n//2)*p for n, p in zip(shape, pitch)]
        q, w = roots_legendre(self.voxel_order)
        offsets = [(np.array([q[i] for i in ids])*pitch/2, np.prod([w[i]/2 for i in ids]))
                   for ids in product(range(self.voxel_order), repeat=3)]
        result = np.zeros(shape)
        for scan, dwell in zip(np.asarray(self.scan_offsets_um), self.dwell_weights):
            if dwell == 0:
                continue
            position = np.asarray(position_um)+scan
            for offset, volume_weight in offsets:
                shifted = [a+v-s for a, v, s in zip(axes, offset, scan)]
                result += dwell*volume_weight*optics.response(shifted, position, wavelength_nm=wavelength_nm,
                                                             order=order, pulse=pulse)
        collection = 1. if emission is None else emission.collection_fraction()
        return result*np.prod(pitch)*self.dwell_seconds*collection
