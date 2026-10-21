"""Composable RANA configuration for physically sampled full-field PSF banks."""
from dataclasses import dataclass, field, asdict
from .physics import VectorialOptics, StratifiedMedium, Acquisition, PulseSpectrum, EmissionResponse
from .depth import DepthPSFBank


@dataclass(frozen=True)
class RanaConfig:
    x_um: tuple
    y_um: tuple
    depths_um: tuple
    voxel_size_um: tuple
    kernel_shape: tuple = (17, 17, 9)
    optics: VectorialOptics = field(default_factory=VectorialOptics)
    acquisition: Acquisition = field(default_factory=Acquisition)
    pulse: PulseSpectrum | None = None
    emission: EmissionResponse | None = None
    wavelength_nm: float = 920.
    photon_order: int = 2

    @classmethod
    def from_dict(cls, specification):
        """Build the same physics configuration from JSON-compatible values."""
        values = dict(specification)
        if "optics" in values:
            optics = dict(values["optics"])
            if "medium" in optics:
                optics["medium"] = StratifiedMedium(**optics["medium"])
            if "polarization" in optics:
                optics["polarization"] = tuple(complex(*v) if isinstance(v, (tuple, list)) else complex(v)
                                                for v in optics["polarization"])
            values["optics"] = VectorialOptics(**optics)
        for name, constructor in (("acquisition", Acquisition), ("pulse", PulseSpectrum),
                                  ("emission", EmissionResponse)):
            if values.get(name) is not None:
                values[name] = constructor(**values[name])
        return cls(**values)

    def build_field(self):
        cache = {}
        def provider(x, y, z):
            # A spatially uniform input pupil has identical XY physics. Cache
            # that exact symmetry; field-dependent pupil callbacks bypass it.
            key = z if self.optics.opd is None and self.optics.amplitude is None else (x, y, z)
            if key not in cache:
                cache[key] = self.acquisition.kernel(self.optics, self.kernel_shape, self.voxel_size_um,
                    (x, y, z), wavelength_nm=self.wavelength_nm, order=self.photon_order,
                    pulse=self.pulse, emission=self.emission)
            return cache[key]
        meta = {'model': 'RANA Richards-Wolf layered pupil', 'medium': asdict(self.optics.medium),
                'vectorial': self.optics.vectorial, 'na': self.optics.na,
                'angular_samples': [self.optics.radial_samples, self.optics.azimuth_samples],
                'reference_wavelength_nm': self.optics.reference_wavelength_nm,
                'polarization': [[v.real, v.imag] for v in self.optics.polarization],
                'pupil_callback': self.optics.opd is not None or self.optics.amplitude is not None,
                'acquisition': asdict(self.acquisition), 'photon_order': self.photon_order,
                'pulse': None if self.pulse is None else asdict(self.pulse),
                'emission': None if self.emission is None else asdict(self.emission),
                'kernel_units': 'relative optical response * um^3 * dwell_seconds'
                                + (' * fs^(1-photon_order)' if self.pulse else '')}
        wavelength = self.wavelength_nm if self.pulse is None else self.pulse.wavelength_nm
        return DepthPSFBank.from_provider(provider, self.x_um, self.y_um, self.depths_um,
            voxel_size_um=self.voxel_size_um, wavelength_nm=wavelength, provenance=meta)


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
