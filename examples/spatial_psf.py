"""A callable scalar-wave provider -> PSF field -> position-dependent images.

This deliberately prescribed wavefront is an interface demonstration, not a
calibrated microscope. Replace wave_provider with your wave-optics program or
load PSFField from NPZ. No RayOptics dependency is needed by this example.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import platform
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maomi.psf import SampledPSF, PSFField, pupil_to_intensity
from maomi.imaging import PSFRenderer
from maomi.synthetic import procedural_cells, noisy_measurement

PITCH = .920/(64*.15*(2/32))  # lambda / (FFT size * NA * normalized pupil step)
EXTENT = 96*PITCH


def wave_provider(x_um, y_um):
    """Paraxial scalar Fourier optics, 920 nm, prescribed position-varying OPD."""
    x, y = np.meshgrid(np.linspace(-1., 1., 33), np.linspace(-1., 1., 33), indexing="ij")
    r2 = x*x+y*y
    fx, fy = 2*x_um/EXTENT-1., 2*y_um/EXTENT-1.
    opd = .40*fx*(x*x-y*y) + .30*fy*(2*r2-1.) + .20*fx*x*(3*r2-2.)
    opd[r2 > 1.] = np.nan
    return SampledPSF(pupil_to_intensity(opd, 64), (PITCH, PITCH), 920.,
                      {"provider": "prescribed scalar Fourier pupil", "NA": .15,
                       "field_um": [x_um, y_um], "calibrated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/spatial_psf"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    nodes = np.linspace(0., EXTENT, 3)
    field = PSFField.from_provider(wave_provider, nodes, nodes)
    provider_seconds = time.perf_counter()-start
    cells = [dict(center_um=[fx*EXTENT, fy*EXTENT], radii_um=[7., 4.],
                  angle_deg=angle, brightness=1., nucleus_fraction=.4)
             for fx, fy, angle in ((.2, .2, 0.), (.5, .3, 60.), (.8, .25, -20.),
                                    (.25, .7, 30.), (.65, .75, -60.))]
    density, components = procedural_cells((96, 96), field.pixel_size_um, cells)
    start = time.perf_counter()
    renderer = PSFRenderer(field, density.shape, pixel_size_um=field.pixel_size_um)
    setup_seconds = time.perf_counter()-start
    start = time.perf_counter(); spatial = renderer(density)
    first_seconds = time.perf_counter()-start
    times = []
    for _ in range(5):
        start = time.perf_counter(); renderer(density); times.append(time.perf_counter()-start)
    center = SampledPSF(field.intensity[1, 1], field.pixel_size_um, 920.)
    common = PSFRenderer(center, density.shape, pixel_size_um=field.pixel_size_um)(density)
    clean = 32*spatial+2.
    noisy = noisy_measurement(clean, rng=np.random.default_rng(891), read_sigma_e=1.)
    field.save(args.output/"psf_field.npz")
    np.savez_compressed(args.output/"images.npz", density=density, cell_components=components,
                        spatial=spatial, common=common, clean_expected_e=clean, noisy_e=noisy)
    result = dict(shape=list(density.shape), grid=[3, 3], active_kernels=renderer.active_kernels,
        pixel_size_um=field.pixel_size_um, cells=cells, noise_seed=891,
        timings_seconds=dict(provider=provider_seconds, renderer_setup=setup_seconds,
            first_render=first_seconds, subsequent_renders=times, warm_median=float(np.median(times))),
        timing_scope="CPU float64, one SciPy FFT worker; setup and provider separate; no noise/I-O/import/plotting",
        environment=dict(python=platform.python_version(), platform=platform.platform()),
        common_psf_relative_l2=float(np.linalg.norm(spatial-common)/np.linalg.norm(spatial)),
        optical_scope="2-D prescribed paraxial scalar wavefront; uncalibrated; separate from 3-D NAOMi scattering")
    (args.output/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for ax, a, title in zip(axes, (density, common, spatial, np.abs(spatial-common)),
                           ("Procedural specimen", "One central PSF", "Spatial PSF field (3 x 3)", "Absolute difference")):
        ax.imshow(a.T, origin="lower", cmap="magma"); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.savefig(args.output/"preview.png", dpi=160); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
