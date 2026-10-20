"""Same cells, activity and acquisition, with optional local-PSF optics."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from maomi import (OpticalConfig, RanaAddon, RanaConfig, VectorialOptics,
                   StratifiedMedium, Acquisition, PulseSpectrum, EmissionResponse,
                   attach_rana, run_fixed)
from maomi.optics import run_optics, locations
from maomi.temporal import ARTraceFilter
from maomi.movie import scan_volume, iter_scan_volume
from maomi.movie_io import write_movie


def inputs(frames=8):
    shape = (24, 24, 12)
    axes = np.meshgrid(*[(np.arange(n)+.5) for n in shape], indexing='ij')
    components, nuclei = [], []
    for center in ((8., 10., 5.), (16., 15., 7.)):
        r2 = sum(((x-c)/r)**2 for x,c,r in zip(axes,center,(3., 2.5, 2.)))
        components.append(np.exp(-2*r2)*(r2 < 2.))
        nuclei.append(r2 < .25)
    gp_vals, gp_nuc = [], []
    for density, nucleus in zip(components,nuclei):
        ids = np.flatnonzero(density.ravel(order='F') > 0)
        gp_vals.append((ids+1, density.ravel(order='F')[ids], np.ones(len(ids),bool)))
        ni = np.flatnonzero(nucleus.ravel(order='F'))
        gp_nuc.append((ni+1, np.ones(len(ni))))
    bg_ids = np.arange(0, np.prod(shape), 7)
    neurons = dict(neur_vol=sum(components), gp_vals=gp_vals, gp_nuc=gp_nuc,
                   bg_proc=[(bg_ids+1, np.full(len(bg_ids), .02))])
    counts = np.zeros((2,frames))
    counts[0,1], counts[1,frames//2] = 1, 1
    soma = ARTraceFilter(2, dt=1/30, tau_decay=.4, tau_rise=.06).process(counts)
    activity = dict(soma=soma, dend=soma, bg=np.full((1,frames), .4))
    motion = [dict(x=3, y=np.full(shape[0],3), z=4+(t%2)) for t in range(frames)]
    cfg = OpticalConfig(volume_um=shape, depth_um=10., tail_um=5., psf_um=(4.,4.,4.))
    return cfg, neurons, activity, motion, counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rana', action='store_true')
    p.add_argument('--physics', action='store_true', help='Attach all four RANA physics extensions')
    p.add_argument('--nuclear', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, default=Path('artifacts/rana_movie'))
    args = p.parse_args(argv)
    if args.physics and not args.rana:
        p.error('--physics requires --rana')
    if args.output.exists():
        p.error('Choose a new output folder')
    torch.set_num_threads(4)
    started = time.perf_counter()
    cfg, neurons, activity, motion, counts = inputs()
    t = time.perf_counter()
    if args.device.startswith('cuda'):
        optics = run_fixed(cfg, neurons['neur_vol'], count_per_axis=3, device=args.device)
    else:
        optics = run_optics(cfg, locations(cfg.volume_um,3), device=args.device)
    propagation_seconds = time.perf_counter()-t
    t = time.perf_counter()
    if args.rana:
        field = None
        if args.physics:
            field = RanaConfig(
                (0.,12.,24.), (0.,12.,24.), (10.,16.,22.), (1.,)*3,
                kernel_shape=(4,4,4),
                optics=VectorialOptics(na=1.1, medium=StratifiedMedium(specimen_index=1.38),
                                      radial_samples=10, azimuth_samples=24),
                acquisition=Acquisition.line((.15,0,0), samples=2, voxel_order=2),
                pulse=PulseSpectrum(fwhm_fs=100., gdd_fs2=1800., samples=33),
                emission=EmissionResponse((500,525,550), (0,1,0), (.5,.8,.6)))
        optics = attach_rana(optics, RanaAddon(field=field, z_origin_um=cfg.depth_um,
                                              device=args.device, status_window=False), config=cfg)
    preparation_seconds = time.perf_counter()-t
    scan = dict(scan_buff=2, scan_avg=1, sfrac=2, nuc_label=int(args.nuclear))
    result = scan_volume(neurons, optics, activity, scan, motion=motion,
                         rng=np.random.default_rng(31), add_noise=True)
    args.output.mkdir(parents=True)
    for name in ('clean','noisy'):
        write_movie((result[name][...,i] for i in range(result['frames'])),
                    args.output/(name+'.npy'), frame_count=result['frames'],
                    shape=result[name].shape[:2], field=name)
    # The streaming path uses exactly the same scanner and motion path.
    stream = iter_scan_volume(neurons, optics, activity, scan, motion=motion, add_noise=False)
    write_movie(stream, args.output/'stream_clean.npy', frame_count=result['frames'],
                shape=result['clean'].shape[:2])
    np.testing.assert_array_equal(np.load(args.output/'stream_clean.npy'),
                                   result['clean'].transpose(2,1,0))
    np.savez_compressed(args.output/'ground_truth.npz', density=neurons['neur_vol'],
                        activity_soma=activity['soma'], spike_counts=counts,
                        motion_z=[m['z'] for m in motion])
    meta = dict(rana=args.rana, physics=args.physics, nuclear=args.nuclear, device=args.device,
                stored_movie_axes='TYX', shape=list(result['clean'].shape),
                clean_target='Noiseless expected acquisition, including background and motion',
                timings=dict(native_propagation=propagation_seconds, rana_preparation=preparation_seconds,
                             movie=result['seconds'], total=time.perf_counter()-started),
                settings=optics.get('settings',{}), stream_matches_collected=True)
    (args.output/'summary.json').write_text(json.dumps(meta, indent=2),encoding='utf8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
