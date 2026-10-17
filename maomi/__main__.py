"""Portable fixed-1024 demo, exporting all computed optical descriptors."""

import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .demo import synthetic_scene
from .fixed import run_fixed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["uniform", "scattering", "vessel"], default="scattering")
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument(
        "--focal-only",
        action="store_true",
        help="Explicitly skip unused C-stage and tail descriptors",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-per-point", action="store_true", help="Also save all 1024 focal PSFs")
    p.add_argument("--output", type=Path, default=Path("artifacts/fixed1024_demo"))
    args = p.parse_args()
    torch.set_num_threads(8)
    jf = Path(str(args.output) + ".json")
    nf = Path(str(args.output) + ".npz")
    if jf.exists() or nf.exists():
        p.error("Output already exists; choose a new --output prefix")
    cfg, volume, vessels = synthetic_scene(args.scene, args.seed)
    try:
        out = run_fixed(
            cfg,
            volume,
            vessels=vessels,
            scatter=args.scene != "uniform",
            tails=not args.focal_only,
            batch_size=args.batch_size,
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
        scope="Optical propagation and noiseless focal static Z-stack",
        per_point_psfs_saved=args.save_per_point,
        timing_note="One-shot time may include JIT/GPU startup; not a warmed benchmark",
    )
    jf.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
