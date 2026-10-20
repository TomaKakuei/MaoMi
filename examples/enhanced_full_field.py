"""MaoMi RANA Mode: complete local excitation PSFs -> clean/noisy image pair.

The built-in 3x3 field is an illustrative effective two-photon response. --field and --density accept your own 2-D or 3-D data.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maomi import FullFieldPSF, simulate
from maomi.rana_demo import rana_demo as demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rana_pairs"))
    parser.add_argument("--field", type=Path, help="FullFieldPSF NPZ; requires --density")
    parser.add_argument("--density", type=Path, help="XY or XYZ nonnegative fluorescence NPY")
    parser.add_argument("--origin-um", type=float, nargs=2, default=(0., 0.))
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--no-status-window", action="store_true", help="Disable the optional status indicator")
    args = parser.parse_args()
    if bool(args.field) != bool(args.density):
        parser.error("Supply both --field and --density, or neither for the demo")
    if args.output.exists():
        parser.error("Choose a new output folder")
    density, field = ((np.load(args.density, allow_pickle=False), FullFieldPSF.load(args.field))
                      if args.field else demo())
    result = simulate(density, mode="rana", psf_field=field,
                      voxel_size_um=field.voxel_size_um, origin_um=args.origin_um,
                      device=args.device, status_window=not args.no_status_window)
    signal = result["image"]
    # Explicit, identical exposure calibration everywhere; retain field-dependent gain.
    clean = signal * 16. + 2.
    rng = np.random.default_rng(20261019)
    noisy = rng.poisson(clean).astype(float) + rng.normal(0, 1., clean.shape)
    args.output.mkdir(parents=True)
    field.save(args.output / "full_field.npz")
    for name, array in [("density", density), ("clean_e", clean), ("noisy_e", noisy)]:
        np.save(args.output / (name + ".npy"), array)
    metadata = {"project": "MaoMi", "route": "RANA Mode", "operator": field.operator,
                "shape": list(density.shape), "field_nodes": list(field.kernels.shape[:2]),
                "voxel_size_um": field.voxel_size_um, "origin_um": args.origin_um,
                "device": args.device, "preparation_seconds": result["timing_seconds"]["preparation"],
                "render_seconds": result["timing_seconds"]["render"], "timing_scope": "single example, not a benchmark",
                "validation": result["validation"], "status_window": result["status_window"], "exposure_e_per_response_unit": 16.,
                "background_e": 2., "read_sigma_e": 1., "noise_seed": 20261019,
                "denoising_target": "clean_e: expected measurement including background",
                "reconstruction_target": "density: latent fluorescence",
                "accuracy_description": "Theoretically higher accuracy through full-field local PSFs"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mid = density.shape[2] // 2 if density.ndim == 3 else None
    view = lambda a: a if mid is None else a[:, :, mid]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, a, title in zip(axes, [density, clean, noisy],
                             ["Latent fluorescence", "Clean expected image (GT)", "Noisy measurement"]):
        im = ax.imshow(view(a).T, origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        fig.colorbar(im, ax=ax, shrink=.72)
    fig.suptitle("MaoMi RANA Mode | local PSF at every position | " + field.operator)
    fig.savefig(args.output / "preview.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
