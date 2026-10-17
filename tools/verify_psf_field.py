"""Independently verify a saved PSF-field demo by direct point-source summation.

No production interpolation weights, renderer, FFT or convolution is used for
the reference. The criterion is final image relative L2 < 0.2% for the supplied
interpolated PSF field, not a claim about physical accuracy or grid sufficiency.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT/"artifacts/spatial_psf")
    args = parser.parse_args()
    folder = args.folder
    with np.load(folder/"psf_field.npz", allow_pickle=False) as f:
        kernels, xx, yy, pitch = [f[k] for k in ("intensity", "x_um", "y_um", "pixel_size_um")]
    with np.load(folder/"images.npz", allow_pickle=False) as f:
        density, saved, clean, noisy = [f[k] for k in ("density", "spatial", "clean_expected_e", "noisy_e")]
    summary = json.loads((folder/"summary.json").read_text())
    start = time.perf_counter()
    nx, ny = density.shape; kx, ky = kernels.shape[-2:]
    interp = RegularGridInterpolator((xx, yy), kernels)
    full = np.zeros((nx+kx-1, ny+ky-1))
    count = 0
    for i, j in np.argwhere(density > 0):
        point = [np.clip((i+.5)*pitch[0], xx[0], xx[-1]), np.clip((j+.5)*pitch[1], yy[0], yy[-1])]
        kernel = interp([point])[0]
        full[i:i+kx, j:j+ky] += density[i, j]*kernel
        count += 1
    reference = full[kx//2:kx//2+nx, ky//2:ky//2+ny]
    reference_seconds = time.perf_counter()-start
    error = float(np.linalg.norm(saved-reference)/np.linalg.norm(reference))
    clean_error = float(np.linalg.norm(clean-(32*reference+2))/np.linalg.norm(32*reference+2))
    # Recompute with the current renderer as well, so stale saved output cannot pass.
    from maomi.psf import PSFField
    from maomi.imaging import render_image
    current = render_image(density, PSFField.load(folder/"psf_field.npz"), pixel_size_um=pitch)
    current_error = float(np.linalg.norm(current-reference)/np.linalg.norm(reference))
    rng = np.random.default_rng(summary["noise_seed"])
    replay = rng.poisson(clean).astype(float)+rng.normal(0., 1., size=clean.shape)
    noise_exact = bool(np.array_equal(replay, noisy))
    result = dict(passed=max(error, clean_error, current_error) < .002 and noise_exact,
        criterion_relative_l2=.002, saved_image_relative_l2=error, clean_expected_relative_l2=clean_error,
        current_renderer_relative_l2=current_error, noise_replay_exact=noise_exact,
        direct_point_sources=count, independent_reference_seconds=reference_seconds,
        reference="SciPy RegularGridInterpolator + explicit point-source scatter; no FFT/convolution",
        scope="Numerical implementation for this supplied PSF field; not grid-convergence or biological accuracy",
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                       for p in ("maomi/psf.py", "maomi/imaging.py", "maomi/synthetic.py", "examples/spatial_psf.py")})
    (folder/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
