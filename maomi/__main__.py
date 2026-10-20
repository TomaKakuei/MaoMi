"""MaoMi: MATLAB-compatible optics with the optional RANA optical add-on."""

import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .demo import synthetic_scene
from .runtime import simulate


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--mode", choices=["standard", "rana"], default="standard",
                          help="standard: MATLAB-compatible; rana: complete custom local PSFs")
    selected, _ = selector.parse_known_args(argv)
    p = argparse.ArgumentParser(description=__doc__, parents=[selector])
    if selected.mode == "rana":
        return rana_main(p, argv)
    p.add_argument("--scene", choices=["uniform", "scattering", "vessel"], default="scattering")
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument(
        "--focal-only",
        action="store_true",
        help="Explicitly skip unused C-stage and tail descriptors",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--count-per-axis", type=int, default=65, help="Optical sites per axis (default: 65, or 4225 sites)")
    p.add_argument("--save-per-point", action="store_true", help="Also save every focal PSF")
    p.add_argument("--output", type=Path, default=Path("artifacts/native_demo"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--rana", action="store_true", help="Enable local PSFs inside the native pipeline")
    p.add_argument("--no-status-window", action="store_true", help="Disable the optional desktop status indicator")
    args = p.parse_args(argv)
    torch.set_num_threads(8)
    jf = Path(str(args.output) + ".json")
    nf = Path(str(args.output) + ".npz")
    if jf.exists() or nf.exists():
        p.error("Output already exists; choose a new --output prefix")
    cfg, volume, vessels = synthetic_scene(args.scene, args.seed)
    from .rana_addon import RanaAddon
    try:
        out = simulate(
            volume, config=cfg, mode="standard", device=args.device,
            vessels=vessels,
            scatter=args.scene != "uniform",
            tails=not args.focal_only,
            batch_size=args.batch_size,
            count_per_axis=args.count_per_axis,
            rana=RanaAddon(status_window=not args.no_status_window) if args.rana else False,
        )
    except (RuntimeError, ValueError) as exc:
        p.error(str(exc))
    arrays = {
        k: out[k]
        for k in ["image", "psf", "psf_raw", "points_um", "energies", "mask", "mask_raw", "colmask"]
    }
    arrays["volume"] = volume
    if args.save_per_point:
        arrays["psf_3d"] = out["psf_3d"]
    for key in ["psfT", "psfB"]:
        for name, value in out.get(key, {}).items():
            arrays[key + "_" + name] = np.asarray(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nf, **arrays)
    metadata = {
        k: out[k]
        for k in [
            "unique_optical_paths",
            "mask_mode",
            "mask_mean",
            "tails_computed",
            "negative_resampling_mass_fraction",
            "timing_seconds",
            "total_seconds",
            "settings",
            "geometry_cache_hit",
        ]
    }
    metadata.update(
        config=asdict(cfg),
        mode="standard",
        scope="Optical propagation and noiseless focal static Z-stack",
        per_point_psfs_saved=args.save_per_point,
        timing_note="One-shot time may include JIT/GPU startup; not a warmed benchmark",
    )
    jf.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(json.dumps(metadata, indent=2))


def rana_main(p, argv):
    from .enhanced import FullFieldPSF
    from .rana_depth import DepthPSFBank
    p.add_argument("--demo", action="store_true", help="Run a self-contained full-field Rana example")
    p.add_argument("--psf-field", type=Path, help="FullFieldPSF NPZ with full XY coverage")
    p.add_argument("--volume", type=Path, help="Nonnegative XY/XYZ fluorescence NPY")
    p.add_argument("--origin-um", nargs=2, type=float, default=(0., 0.))
    p.add_argument("--z-origin-um", type=float, help="Lower Z edge; defaults to the first bank depth")
    p.add_argument("--physics-demo", action="store_true", help="Layered vectorial optics, voxel/scan integration and coherent pulse spectrum")
    p.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    p.add_argument("--no-status-window", action="store_true", help="Disable the optional desktop status indicator")
    p.add_argument("--output", type=Path, default=Path("artifacts/rana_demo"))
    args = p.parse_args(argv)
    config = None
    if args.demo or args.physics_demo:
        if args.demo and args.physics_demo:
            p.error("Choose --demo or --physics-demo")
        if args.psf_field or args.volume:
            p.error("Choose --demo or --psf-field plus --volume")
        from .rana_demo import rana_demo, physics_demo
        if args.physics_demo:
            volume, config = physics_demo()
            field = None
        else:
            volume, field = rana_demo()
    else:
        if not args.psf_field or not args.volume:
            p.error("Rana Mode requires --psf-field and --volume, or --demo")
        try:
            volume = np.load(args.volume, allow_pickle=False)
            with np.load(args.psf_field, allow_pickle=False) as data:
                depth_bank = str(data['schema']) == 'maomi.depth_psf_bank.v1'
            field = (DepthPSFBank if depth_bank else FullFieldPSF).load(args.psf_field)
        except (OSError, ValueError, KeyError) as exc:
            p.error(str(exc))
    jf, nf = Path(str(args.output)+".json"), Path(str(args.output)+".npz")
    if jf.exists() or nf.exists():
        p.error("Output already exists; choose a new --output prefix")
    try:
        z_origin = args.z_origin_um
        if z_origin is None:
            z_origin = config.depths_um[0] if config is not None else field.depths_um[0] if isinstance(field, DepthPSFBank) else 0.
        out = simulate(volume, mode="rana", psf_field=field, config=config,
                       voxel_size_um=field.voxel_size_um if field is not None else config.voxel_size_um,
                       origin_um=args.origin_um, z_origin_um=z_origin, device=args.device,
                       status_window=not args.no_status_window)
    except (RuntimeError, ValueError, TypeError) as exc:
        p.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nf, image=out["image"], volume=volume)
    metadata = {k: v for k, v in out.items() if k != "image"}
    metadata.update(project="MaoMi", display_mode="RANA - Ray Aware Nonuniform Acquisition",
                    input="built-in physics" if args.physics_demo else "built-in custom field" if args.demo else "user supplied field and volume",
                    shape=list(volume.shape), wavelength_nm=field.wavelength_nm if field is not None else config.wavelength_nm,
                    scope="Noiseless full-field static image; exposure and noise are separate",
                    accuracy_description="Theoretically higher accuracy through full-field local PSFs")
    jf.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
