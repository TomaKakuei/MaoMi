"""512-pixel moving-cell comparison using the actual shared movie workflow.

Run: python examples/timed_acquisition.py --output artifacts/timing --device cpu
Creates a figure, PDF and numerical summary; no large movie files are retained.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from maomi import Scene, OpticalConfig, FullFieldPSF, RanaAddon, RunConfig, run_scene
from maomi.acquisition import AcquisitionClock
from maomi.comparison import compare_images


def make_scene():
    n = 512
    x, y = np.indices((n, n), dtype=float)
    density = np.zeros((n, n), float)
    # Fixed procedural geometry: no external photographs or fitted specimen.
    for cx, cy, rx, ry in ((155, 120, 20, 31), (340, 130, 22, 28),
                            (248, 253, 23, 40), (370, 285, 19, 30),
                            (165, 382, 25, 34), (341, 401, 22, 26)):
        radius = np.sqrt(((x-cx)/rx)**2+((y-cy)/ry)**2)
        density += np.exp(-((radius-1)/.12)**2)+.45*np.exp(-radius**2/.12)
    density[density < 1e-7] = 0.
    volume = density[..., None]
    ids = np.flatnonzero(volume.ravel(order="F"))
    neurons = dict(neur_vol=volume, gp_vals=[(ids+1, volume.ravel(order="F")[ids], np.ones(len(ids), bool))], bg_proc=[])
    scene = Scene(neurons, OpticalConfig(volume_um=(n, n, 1), psf_um=(11, 11, 1)),
                  activity=dict(soma=np.ones((1, 1))), spikes=dict(dt=.05),
                  scan=dict(scan_buff=0, sfrac=1, scan_avg=1))
    kx, ky = np.meshgrid(np.arange(11)-5., np.arange(11)-5., indexing="ij")
    kernels = np.empty((2, 2, 11, 11, 1))
    for i, j in np.ndindex(2, 2):
        k = np.exp(-.5*((kx/(1.0+i*.8))**2+(ky/(1.0+j*.8))**2))
        kernels[i, j, ..., 0] = k/k.sum()
    field = FullFieldPSF(kernels, [0., n], [0., n], (1.,)*3, 920., "excitation",
                         {"source": "analytic spatial Gaussian excitation kernels"})
    optics = dict(psf=kernels[0, 0])
    return scene, optics, field


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene, optics, field = make_scene()
    addon = RanaAddon(field=field, device=args.device, status_window=False)
    settings = RunConfig(device=args.device, add_noise=False)
    clock = AcquisitionClock("raster", 1e-7, 8e-5, 1e-7, samples=513)
    clocks = (AcquisitionClock("global", 1e-7, samples=3), clock, replace(clock, reverse=True))
    images, timings = [], []
    def render(c):
        from dataclasses import asdict
        scene.scan["timing"] = dict(asdict(c), velocity_voxels_s=[4000., 0., 0.], photon_rate_scale=1e9)
        return run_scene(scene, optics=optics, rana=addon, settings=settings)["clean"][..., 0]
    for label, c in zip(("snapshot", "forward", "reverse"), clocks):
        start = time.perf_counter()
        images.append(render(c))
        timings.append(time.perf_counter()-start)
        print(f"{label}: {timings[-1]:.3f} s", flush=True)
    refined = render(replace(clock, samples=1025))
    roi = ((188, 308), (185, 321))
    metrics = dict(resolution=[512, 512], pitch_um=1., frame_midpoint_s=0.,
                   raster_span_s=float(np.ptp(clock.offsets((512, 512)))),
                   velocity_um_s=[4000., 0., 0.], exposure_s=clock.exposure_s,
                   samples=clock.samples, fine_samples=1025,
                   forward=compare_images(images[0], images[1], roi=roi),
                   reverse=compare_images(images[0], images[2], roi=roi),
                   convergence=compare_images(refined, images[1], roi=roi),
                   elapsed_seconds=dict(zip(("snapshot", "forward", "reverse"), timings)),
                   interpretation="same time midpoint, same photons-per-rate scale; snapshot is a simultaneous-sampling control",
                   provenance="synthetic fluorescence; native MaoMi RANA local-PSF movie renderer; no image registration")
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    if metrics["convergence"]["full_field"]["relative_l2"] >= .002:
        raise RuntimeError("Temporal refinement exceeds 0.2%; inspect metrics before publishing")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 15), facecolor="#f6f8fc")
    grid = fig.add_gridspec(3, 3, left=.05, right=.94, bottom=.09, top=.89,
                           height_ratios=[1, .72, .68], wspace=.2, hspace=.28)
    names = ["Simultaneous snapshot", "Raster | forward Y", "Raster | reverse Y"]
    vmax = max(float(im.max()) for im in images)
    for col, (name, im) in enumerate(zip(names, images)):
        ax = fig.add_subplot(grid[0, col])
        artist = ax.imshow(im.T, origin="lower", cmap="magma", vmin=0, vmax=vmax, extent=(0, 512, 0, 512))
        ax.add_patch(Rectangle((roi[0][0], roi[1][0]), roi[0][1]-roi[0][0], roi[1][1]-roi[1][0], fill=False, edgecolor="#56dfd4", lw=1.4))
        ax.set_title(name, loc="left", pad=10)
        ax.set_xlabel("X (um)")
        if col == 0:
            ax.set_ylabel("Y (um)")
        ax = fig.add_subplot(grid[1, col])
        crop = im[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        ax.imshow(crop.T, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                  extent=(*roi[0], *roi[1]), aspect="equal")
        ax.set_title("Same cell / fixed coordinates", fontsize=11, loc="left")
        ax.set_xlabel("X (um)")
        if col == 0:
            ax.set_ylabel("Y (um)")
    cax = fig.add_axes([.952, .62, .011, .20])
    fig.colorbar(artist, cax=cax, label="Expected photons / pixel")
    ax = fig.add_subplot(grid[2, 0])
    yy = np.arange(512)
    for name, c, color in zip(names, clocks, ("#354c76", "#e36542", "#168b8b")):
        ax.plot(1000*c.offsets((512, 512))[256], yy, color=color, label=name)
    ax.set(xlabel="Sample midpoint relative to frame (ms)", ylabel="Y (um)", title="Explicit scan timing")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=.15)
    peak = max(float(np.abs(images[i]-images[0]).max()) for i in (1, 2))
    for col in (1, 2):
        ax = fig.add_subplot(grid[2, col])
        residual = images[col]-images[0]
        diff = ax.imshow(residual.T, origin="lower", cmap="RdBu_r", vmin=-peak, vmax=peak,
                         extent=(0, 512, 0, 512))
        record = metrics["forward" if col == 1 else "reverse"]
        ax.set_title(f"Raster - snapshot | field {record['full_field']['relative_l2']*100:.1f}% L2\n"
                     f"Fixed cell ROI {record['roi']['relative_l2']*100:.1f}% L2", fontsize=10, loc="left")
        ax.set_xlabel("X (um)")
    cax = fig.add_axes([.952, .11, .011, .17])
    fig.colorbar(diff, cax=cax, label="Signed photon difference")
    fig.text(.05, .965, "MaoMi | Motion happens during acquisition", fontsize=25, weight="bold", color="#192c4b")
    fig.text(.05, .927, "512 x 512 fluorescence scene  /  local PSFs  /  same frame midpoint  /  one shared absolute intensity scale", fontsize=12, color="#475875")
    error = metrics["convergence"]["full_field"]["relative_l2"]*100
    fig.text(.05, .037, f"513 -> 1025 temporal samples: {error:.4f}% image L2 change.  "
             "Rigid cells; scan-dependent shape arises before detector noise.\n"
             "Synthetic 4 mm/s flow; 100 ns dwell, 80 us line period, 40.93 ms scan. "
             "Snapshot is a timing control, not a claim of a complete camera model.", fontsize=10, color="#475875")
    fig.savefig(args.output/"timed_acquisition.png", dpi=260, facecolor=fig.get_facecolor())
    fig.savefig(args.output/"timed_acquisition.pdf", facecolor=fig.get_facecolor(), metadata={"Author": "Yueze Wang"})
    plt.close(fig)
    print(json.dumps({"convergence_percent": error, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
