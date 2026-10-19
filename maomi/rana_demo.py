"""Self-contained custom-field input for the explicit Rana Mode demo."""
import numpy as np
from .enhanced import FullFieldPSF


def rana_demo():
    pitch = (.5, .5)
    shape = (192, 192)
    extent = np.array(shape) * pitch
    x, y = np.meshgrid((np.arange(shape[0]) + .5) * pitch[0],
                       (np.arange(shape[1]) + .5) * pitch[1], indexing="ij")
    density = np.zeros(shape)
    rng = np.random.default_rng(20261019)
    for cx, cy, radius, brightness in rng.uniform([8, 8, 1.2, .5], [88, 88, 3, 1.5], (35, 4)):
        density += brightness * np.exp(-((x-cx)**2 + (y-cy)**2)/(2*radius**2))
    kx, ky = np.meshgrid((np.arange(17)-8)*pitch[0], (np.arange(17)-8)*pitch[1], indexing="ij")
    # Square each illustrative one-photon intensity ONCE. Mix effective responses
    # rather than complex amplitudes; this is a prescribed phenomenological field.
    core = np.exp(-.5*((kx/.65)**2 + (ky/.8)**2))**2
    wing_x = np.exp(-.5*(((kx-.6)/1.25)**2 + (ky/.7)**2))**2
    wing_y = np.exp(-.5*((kx/.75)**2 + ((ky+.5)/1.15)**2))**2
    def optics(xx, yy):
        return core + .45 * xx/extent[0] * wing_x + .35 * yy/extent[1] * wing_y
    field = FullFieldPSF.from_provider(optics, np.linspace(0, extent[0], 3),
                                      np.linspace(0, extent[1], 3), voxel_size_um=pitch,
                                      wavelength_nm=920., operator="excitation", provenance={
                                          "provider": "illustrative affine mixture of effective two-photon responses",
                                          "units": "relative response, fixed demo exposure",
                                          "calibrated_instrument": False})
    return density, field

def physics_demo():
    """A complete 128 x 128 x 9 RANA volume with all four physics extensions."""
    from .rana import RanaConfig
    from .rana_optics import VectorialOptics, StratifiedMedium, Acquisition, PulseSpectrum, EmissionResponse
    import numpy as np
    shape, pitch = (128, 128, 9), (.2, .2, .5)
    x, y, z = np.meshgrid(*[(np.arange(n)+.5)*p for n, p in zip(shape, pitch)], indexing='ij')
    volume = np.zeros(shape)
    rng = np.random.default_rng(20261019)
    for _ in range(12):
        cx, cy, cz = rng.uniform([2, 2, .3], [23, 23, 4.2])
        radius = rng.uniform(.5, 1.2)
        volume += np.exp(-((x-cx)**2+(y-cy)**2+(z-cz)**2)/(2*radius**2))
    volume += .4*np.exp(-((y-12-2*np.sin(x/3))/.25)**2-((z-2)/.5)**2)
    cfg = RanaConfig((0., 12.8, 25.6), (0., 12.8, 25.6), (40., 42.25, 44.5), pitch,
        optics=VectorialOptics(na=1.1, medium=StratifiedMedium(specimen_index=1.38),
                              radial_samples=16, azimuth_samples=40),
        acquisition=Acquisition.line((.18, 0., 0.), samples=2, voxel_order=2),
        pulse=PulseSpectrum(fwhm_fs=100, gdd_fs2=1800, samples=33),
        emission=EmissionResponse((500, 525, 550, 575), (0., 1., .7, 0.), (.55, .8, .7, .3)))
    return volume, cfg
