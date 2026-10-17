"""Run a real RayOptics -> MAOMi -> paired training-data demonstration on CPU.

python examples/rayoptics_custom_psf.py --output artifacts/rayoptics_demo
python examples/rayoptics_custom_psf.py --cells examples/cell_recipe.json --output artifacts/my_cells
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from maomi.integrations import SampledPSF
from maomi.integrations.rayoptics import demo_lens, rayoptics_psf
from maomi.synthetic import procedural_cells, expected_measurement, noisy_measurement


def make_cells(rng, extent):
    cells = []
    for _ in range(5):
        c = rng.uniform(.18, .82, 2)*extent
        angle = rng.uniform(0., 2*np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        end = np.clip(c+direction*rng.uniform(12., 23.), 2., extent-2.)
        middle = .5*(c+end)+rng.normal(0., 2., 2)
        cells.append(dict(center_um=c.tolist(), radii_um=rng.uniform(3., 7., 2).tolist(),
            angle_deg=float(rng.uniform(0., 180.)), brightness=float(rng.uniform(.6, 1.3)),
            nucleus_fraction=float(rng.uniform(.2, .65)), branch_width_um=float(rng.uniform(.6, 1.2)),
            branches=[[c.tolist(), middle.tolist(), end.tolist()]]))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', type=Path, help='Custom cell recipe JSON; generates one scene without a training split')
    parser.add_argument('--psf', type=Path, help='Load a provider-neutral SampledPSF NPZ instead of tracing the demo lens')
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--pupil-mm', type=float, default=8.)
    parser.add_argument('--wavelength-nm', type=float, default=550.)
    parser.add_argument('--focus-offset-mm', type=float, default=0.)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory to preserve existing evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    if args.psf:
        psf = SampledPSF.load(args.psf)
    else:
        psf = rayoptics_psf(demo_lens(wavelength_nm=args.wavelength_nm, entrance_pupil_mm=args.pupil_mm),
                           focus_offset=args.focus_offset_mm)
        psf.provenance['prescription'] = dict(object_distance_mm=100., glass_index=1.517,
            first_radius_mm=50., second_radius_mm=-50., thickness_mm=4.,
            entrance_pupil_mm=args.pupil_mm, image_distance='paraxial focus + focus_offset')
    optics_seconds = time.perf_counter()-started
    psf.save(args.output/'psf.npz')
    shape = (96, 96)
    extent = np.asarray(shape)*psf.pixel_size_um
    custom = json.loads(args.cells.read_text(encoding='utf-8')) if args.cells else None
    nscenes = 1 if custom is not None else 9
    # Split whole scenes: all four noise/dose variants remain together.
    splits = ['custom'] if custom is not None else ['train']*6+['validation']+['test']*2
    root_seed = np.random.SeedSequence(args.seed)
    scene_seeds = root_seed.spawn(nscenes)
    densities, clean, noisy, components, recipes, pairs = [], [], [], [], [], []
    data_started = time.perf_counter()
    for scene_id, seed in enumerate(scene_seeds):
        geometry_seed, *noise_seeds = seed.spawn(5)
        cells = custom['cells'] if custom is not None else make_cells(np.random.default_rng(geometry_seed), extent)
        density, layers = procedural_cells(shape, psf.pixel_size_um, cells)
        densities.append(density)
        components.append(layers)
        recipes.append(dict(scene_id=scene_id, split=splits[scene_id], cells=cells,
                            geometry_seed=dict(entropy=args.seed, spawn_key=list(geometry_seed.spawn_key))))
        for dose, noise_seed in zip([2., 8., 32., 128.], noise_seeds):
            expected = expected_measurement(density, psf, pixel_size_um=psf.pixel_size_um,
                                            photons_per_unit=dose, background_e=2.)
            observed = noisy_measurement(expected, rng=np.random.default_rng(noise_seed), read_sigma_e=1.)
            clean.append(expected)
            noisy.append(observed)
            pairs.append(dict(pair_id=len(pairs), scene_id=scene_id, split=splits[scene_id],
                detected_e_per_density_unit=dose, background_e=2., read_sigma_e=1.,
                noise_seed=dict(entropy=args.seed, spawn_key=list(noise_seed.spawn_key))))
    generation_seconds = time.perf_counter()-data_started
    clean, noisy, densities = map(np.asarray, (clean, noisy, densities))
    save_started = time.perf_counter()
    np.savez_compressed(args.output/'pairs.npz', noisy_e=noisy, clean_expected_e=clean,
        latent_density=densities, cell_components=np.asarray(components),
        pair_scene_id=[p['scene_id'] for p in pairs], split=[p['split'] for p in pairs],
        detected_e_per_density_unit=[p['detected_e_per_density_unit'] for p in pairs],
        pixel_size_um=psf.pixel_size_um, axes='NXY')
    data_save_seconds = time.perf_counter()-save_started
    manifest = dict(schema='maomi.paired_thin_section.v1', seed=args.seed, shape_xy=shape,
        axes='XY; arrays have a leading sample/scene dimension', pixel_size_um=psf.pixel_size_um,
        density_unit='relative fluorescence density', measurement_unit='detected photoelectrons',
        denoising_target='clean_expected_e: expected measurement including blur and background',
        latent_target='latent_density: ideal specimen density; a separate deconvolution task',
        detector='Poisson(clean_expected_e) + Normal(0, read_sigma_e); no clipping or quantization',
        boundary='zero-extended MAOMi zstack convolution; light may leave the image',
        split_policy='scene-disjoint; all dose/noise variants of a scene have one split',
        psf=psf.provenance, pairs=pairs, scenes=recipes,
        files={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['psf.npz', 'pairs.npz']})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    checks = dict(psf_unit_sum=bool(abs(psf.intensity.sum()-1.)<1e-12),
        finite_arrays=bool(all(np.isfinite(a).all() for a in [densities, clean, noisy])),
        nonnegative_targets=bool(densities.min()>=0 and clean.min()>=0),
        component_additivity=bool(np.allclose(np.asarray(components).sum(axis=1), densities)),
        scene_disjoint_splits=all(len({p['split'] for p in pairs if p['scene_id']==i})==1 for i in range(nscenes)))
    summary = dict(passed=all(checks.values()), checks=checks, scenes=nscenes, paired_images=len(pairs),
        image_shape_xy=shape, pixel_size_um=psf.pixel_size_um, psf_shape_xy=psf.intensity.shape,
        valid_traced_rays=psf.provenance.get('valid_rays'), seed=args.seed,
        timings_seconds=dict(optics_provider=optics_seconds, cell_render_and_pair_generation=generation_seconds,
            compressed_array_export=data_save_seconds, total_through_manifest=time.perf_counter()-started),
        timing_scope='one CPU invocation, 4 Torch threads; excludes interpreter/import startup and preview plotting; not a MATLAB/Julia benchmark',
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            rayoptics=importlib.metadata.version('rayoptics') if not args.psf else 'external PSF'),
        limitations=['Stylized 2-D cells; no real-data calibration or denoiser training.',
                     'Scalar, central-wavelength, near-axis optics; no vector high-NA or tissue-scattering composition.',
                     'A single optical prescription and noise family do not demonstrate transfer to real microscopes.'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    render_preview(psf, densities, clean, noisy, args.output/'preview.png')
    print(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise SystemExit(1)


def render_preview(psf, densities, clean, noisy, path):
    """Render saved arrays; plotting is excluded from the recorded execution time."""
    # Display transposes XY to conventional horizontal X / vertical Y.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.15), constrained_layout=True)
    half = 25
    mid = np.asarray(psf.intensity.shape)//2
    patch = psf.intensity[mid[0]-half:mid[0]+half+1, mid[1]-half:mid[1]+half+1]
    views = [densities[0], patch, clean[3], noisy[3]]
    titles = ['Procedural cells', 'Ray-traced PSF (crop)', 'Expected measurement', 'Noisy measurement']
    for ax, arr, title in zip(axs, views, titles):
        im = ax.imshow(arr.T, origin='lower', cmap='magma', vmin=0.,
                       vmax=float(clean[3].max()) if 'measurement' in title.lower() else None)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
    fig.suptitle('Editable cells + RayOptics lens + MAOMi acquisition (128 detected electrons / density unit)', fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
