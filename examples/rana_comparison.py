"""Reproduce RANA physics ablations, quadrature convergence and serial timings.

Outputs are local artifacts. Only the small aggregate table/figure is published.
No MATLAB execution is needed; its separate native benchmark remains unchanged.
"""
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
import sys
import time
import platform
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maomi import (RanaConfig, VectorialOptics, StratifiedMedium, Acquisition, PulseSpectrum,
                   EmissionResponse, DepthPSFBank, FullFieldPSF, DepthFullFieldRenderer)


def scene():
    shape, pitch = (256, 256, 16), (.15, .15, .75)
    x, y, z = np.meshgrid(*[(np.arange(n)+.5)*p for n, p in zip(shape, pitch)], indexing='ij')
    volume = np.zeros(shape)
    rng = np.random.default_rng(20261019)
    for _ in range(32):
        cx, cy, cz = rng.uniform([2, 2, 1], [36, 36, 11])
        radius = rng.uniform(.55, 1.4)
        volume += rng.uniform(.5, 1)*np.exp(-((x-cx)**2+(y-cy)**2+(z-cz)**2)/(2*radius**2))
    for intercept in (8, 18, 29):
        volume += .5*np.exp(-((y-intercept-2*np.sin(x/3))/.22)**2-((z-6)/.7)**2)
    return volume, pitch


def scaled_bank(bank, factor):
    return DepthPSFBank(bank.depths_um, tuple(FullFieldPSF(f.kernels*factor, f.x_um, f.y_um,
        f.voxel_size_um, f.wavelength_nm, f.operator, f.provenance) for f in bank.fields))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--reference-cache', type=Path, help='Reuse a completed numerical reference with an identical physical protocol')
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output directory; prior runs are preserved')
    args.output.mkdir(parents=True)
    volume, pitch = scene()
    optics = VectorialOptics(na=1.1, medium=StratifiedMedium(specimen_index=1.38),
                            radial_samples=16, azimuth_samples=40)
    pulse = PulseSpectrum(gdd_fs2=1800, samples=33)
    cfg = RanaConfig((0, 19.2, 38.4), (0, 19.2, 38.4), (50, 56, 62), pitch,
        kernel_shape=(21, 21, 9), optics=optics,
        acquisition=Acquisition.line((.18, 0, 0), samples=2, voxel_order=3), pulse=pulse,
        emission=EmissionResponse((500, 525, 550, 575), (0, 1, .7, 0), (.55, .8, .7, .3)))
    settings = {'shape': list(volume.shape), 'voxel_size_um': pitch, 'seed': 20261019,
        'na': optics.na, 'medium': asdict(optics.medium), 'depth_nodes_um': list(cfg.depths_um),
        'kernel_shape': cfg.kernel_shape, 'angular_samples': [16, 40],
        'acquisition': asdict(cfg.acquisition), 'pulse': asdict(pulse), 'emission': asdict(cfg.emission),
        'reference': {'depth_nodes_um': [50, 53, 56, 59, 62], 'angular_samples': [24, 64],
                      'voxel_order': 4, 'scan_samples': 3, 'spectral_samples': 49},
        'execution': 'Serial; CPU PSF construction; selected-device FFT rendering. No I/O in measured stages.',
        'device': args.device, 'python': platform.python_version(), 'platform': platform.platform(),
        'timing': 'One fresh bank build per route. Render speed: median of three warm calls on the identical bank.',
        'image_metric': '100 * norm(route-reference) / norm(reference), full noiseless XYZ volume; no intensity fitting',
        'comparison_scope': 'Prescribed layered optical model: ablation effects and numerical convergence, not a MATLAB equivalence test'}
    (args.output/'protocol.json').write_text(json.dumps(settings, indent=2), encoding='utf8')
    images, records = {}, {}
    def run(name, config=None, bank=None, factor=1.):
        started = time.perf_counter()
        if bank is None:
            bank = config.build_field()
            if factor != 1:
                bank = scaled_bank(bank, factor)
        built = time.perf_counter()
        bank.save(args.output/(name+'_bank.npz'))
        # Serialization is outside the measured preparation interval.
        before_prepare = time.perf_counter()
        renderer = DepthFullFieldRenderer(bank, volume.shape, voxel_size_um=pitch, z_origin_um=50, device=args.device)
        prepared = time.perf_counter()
        image = renderer(volume)
        done = time.perf_counter()
        records[name] = {'psf_build_seconds': built-started, 'prepare_seconds': prepared-before_prepare,
                         'render_seconds': done-prepared, 'total_seconds': built-started+done-before_prepare,
                         'bank_reused': config is None}
        images[name] = image
        np.save(args.output/(name+'.npy'), image)
        (args.output/'progress.json').write_text(json.dumps(records, indent=2), encoding='utf8')
        print(name, json.dumps(records[name]), flush=True)
        return bank
    bank = run('rana', config=cfg)
    run('fixed_depth', bank=DepthPSFBank(bank.depths_um, (bank.fields[1],)*3))
    scalar = VectorialOptics(na=optics.na, medium=optics.medium, vectorial=False, radial_samples=16, azimuth_samples=40)
    run('scalar', config=replace(cfg, optics=scalar))
    run('point_sampling', config=replace(cfg, acquisition=Acquisition(voxel_order=1)))
    temporal_scale = np.sqrt(2*np.log(2)/np.pi)/pulse.fwhm_fs
    run('monochromatic_transform_limited', config=replace(cfg, pulse=None), factor=temporal_scale)
    simple_cfg = replace(cfg, optics=scalar, pulse=None, acquisition=Acquisition(voxel_order=1), depths_um=(50, 62))
    started = time.perf_counter()
    k = simple_cfg.acquisition.kernel(scalar, cfg.kernel_shape, pitch, (0, 0, 56), emission=cfg.emission)*temporal_scale
    f = FullFieldPSF(np.broadcast_to(k, (3, 3, *k.shape)), cfg.x_um, cfg.y_um, pitch, 920, 'excitation')
    simple_bank = DepthPSFBank((50, 62), (f, f))
    simple_build = time.perf_counter()-started
    run('simplified', bank=simple_bank)
    records['simplified']['psf_build_seconds'] = simple_build
    records['simplified']['total_seconds'] += simple_build
    records['simplified']['bank_reused'] = False
    reference = replace(cfg, depths_um=(50, 53, 56, 59, 62),
        optics=VectorialOptics(na=optics.na, medium=optics.medium, radial_samples=24, azimuth_samples=64),
        acquisition=Acquisition.line((.18, 0, 0), samples=3, voxel_order=4), pulse=replace(pulse, samples=49))
    if args.reference_cache:
        previous = json.loads((args.reference_cache/'summary.json').read_text(encoding='utf8'))
        physics_keys = ('shape', 'voxel_size_um', 'seed', 'na', 'medium', 'depth_nodes_um', 'kernel_shape',
                        'angular_samples', 'acquisition', 'pulse', 'emission', 'reference')
        normalized = json.loads(json.dumps(settings))
        if any(previous['protocol'][k] != normalized[k] for k in physics_keys):
            raise ValueError('Cached reference has a different physical protocol')
        source = args.reference_cache/'reference.npy'
        image = np.load(source, allow_pickle=False)
        if image.shape != volume.shape or not np.isfinite(image).all() or (image < 0).any():
            raise ValueError('Invalid cached reference image')
        images['reference'] = image
        np.save(args.output/'reference.npy', image)
        records['reference'] = {**previous['routes']['reference'], 'reused_reference': True,
                                'reference_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    else:
        run('reference', config=reference)
    norm = lambda x: float(np.linalg.norm(x.ravel()))
    for name, entry in records.items():
        entry['relative_l2_percent_vs_rana'] = 100*norm(images[name]-images['rana'])/norm(images['rana'])
        entry['relative_l2_percent_vs_reference'] = 100*norm(images[name]-images['reference'])/norm(images['reference'])
    render_benchmark = {}
    for device in (['cpu', 'cuda'] if args.device == 'cuda' else ['cpu']):
        renderer = DepthFullFieldRenderer(bank, volume.shape, voxel_size_um=pitch, z_origin_um=50, device=device)
        output = renderer(volume)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            output = renderer(volume)
            times.append(time.perf_counter()-start)
        render_benchmark[device] = {'seconds': times, 'median_seconds': float(np.median(times)),
            'relative_l2_percent_vs_rana': 100*norm(output-images['rana'])/norm(images['rana'])}
    report = {'protocol': settings, 'routes': records, 'render_benchmark': render_benchmark,
              'collection_fraction': cfg.emission.collection_fraction(),
              'broadened_pulse_fwhm_fs': pulse.broadened_fwhm_fs}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    np.save(args.output/'density.npy', volume)
    figure(images, volume, args.output/'comparison.png', report)
    print(json.dumps(report, indent=2), flush=True)


def figure(images, volume, path, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    sl = 8
    maximum = max(images['rana'][:, :, sl].max(), images['simplified'][:, :, sl].max())
    panels = [('Density / synthetic GT', volume[:, :, sl], None),
              ('Simplified optical model', images['simplified'][:, :, sl], maximum),
              ('RANA / all four extensions', images['rana'][:, :, sl], maximum),
              ('Higher quadrature reference', images['reference'][:, :, sl], maximum),
              ('RANA - simplified', images['rana'][:, :, sl]-images['simplified'][:, :, sl], 'signed'),
              ('RANA - reference', images['rana'][:, :, sl]-images['reference'][:, :, sl], 'signed')]
    for ax, (title, data, scale) in zip(axes.ravel(), panels):
        if scale == 'signed':
            bound = max(abs(data.min()), abs(data.max()))
            im = ax.imshow(data.T, origin='lower', extent=(0, 38.4, 0, 38.4), cmap='RdBu_r', vmin=-bound, vmax=bound)
        else:
            im = ax.imshow(data.T, origin='lower', extent=(0, 38.4, 0, 38.4), cmap='magma', vmin=0, vmax=scale)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')
        fig.colorbar(im, ax=ax, fraction=.045)
    error = report['routes']['rana']['relative_l2_percent_vs_reference']
    fig.suptitle(f'RANA | 256 x 256 x 16 volume | full-volume convergence difference {error:.4f}%\n'
                 'Same density and exposure; image panels share a scale; signed differences use their own labeled scales', fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
