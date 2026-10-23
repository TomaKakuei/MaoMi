"""Ideal component images and explicitly scoped detector SNR estimates."""

import numpy as np
from scipy import ndimage
from .movie import scan_volume


def ideal_support(components, baseline, *, k=2.0, minimum_pixels=5, mode="stable"):
    """Original comps2ideals support rule, with empty/zero-background handling.

    This is a spatial support heuristic, not a calibrated detector SNR value.
    Stable mode excludes baseline values below 64 float32 eps of its peak:
    FFT roundoff in dark pixels must not set the global component threshold.
    Legacy mode reproduces the original ratio rule, including that instability.
    """
    components = np.asarray(components)
    baseline = np.asarray(baseline)
    if (
        components.ndim != 3
        or baseline.shape != components.shape[:2]
        or not np.isfinite(components).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("Finite XYK components and XY baseline required")
    if minimum_pixels < 1 or int(minimum_pixels) != minimum_pixels or k < 0:
        raise ValueError("Invalid support parameters")
    result = np.zeros_like(components)
    if mode not in ("stable", "legacy"):
        raise ValueError("Unknown support mode")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = components.astype(float) / baseline[:, :, None]
    if mode == "stable":
        valid = baseline > 64 * np.finfo(np.float32).eps * np.max(np.abs(baseline))
        ratios = np.where(valid[:, :, None], ratios, 0.0)
    for j in range(components.shape[2]):
        ratio = ratios[:, :, j]
        values = np.sort(ratio[~np.isnan(ratio)])[::-1]
        if len(values) < minimum_pixels:
            continue
        mean = float(values[:minimum_pixels].mean())
        if mean <= 0:
            continue
        cutoff = 1 / (k + 2 / mean)
        mask = ratio > cutoff
        labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
        if not n:
            continue
        sizes = np.bincount(labels.ravel())[1:]
        candidates = np.flatnonzero(sizes == sizes.max()) + 1
        chosen = min(candidates, key=lambda x: np.flatnonzero((labels == x).ravel(order="F"))[0])
        if sizes[chosen - 1] >= minimum_pixels:
            result[:, :, j] = np.where(labels == chosen, components[:, :, j], 0)
    return result


def ideal_components(
    neurons, optics, activity, scan=None, *, spikes=None, tpm=None, grouping="by_cell"
):
    """Render one selected baseline component at a time and the all-on baseline.

    by_cell follows original calculateIdealComps when compartment counts match.
    by_compartment permits an independent neuropil count and returns explicit
    (compartment,index) labels. No claim that overlaps add linearly is made.
    """
    arrays = {
        key: np.atleast_2d(activity[key]).astype(np.float32) for key in ("soma", "dend", "bg")
    }
    mins = {k: a.min(axis=1) for k, a in arrays.items()}
    counts = {k: len(a) for k, a in arrays.items()}
    if grouping == "by_cell":
        if len(set(counts.values())) != 1:
            raise ValueError("Use by_compartment for unequal component counts")
        count = next(iter(counts.values()))
        labels = [("cell", i) for i in range(count)]
        active = [i for i in range(count) if sum(mins[k][i] for k in mins) > 0]
        selected = {k: np.zeros((count, len(active)), np.float32) for k in arrays}
        for col, j in enumerate(active):
            for k in arrays:
                selected[k][j, col] = mins[k][j]
    elif grouping == "by_compartment":
        labels = [(k, i) for k in arrays for i in range(counts[k])]
        active = [i for i, (k, j) in enumerate(labels) if mins[k][j] > 0]
        selected = {k: np.zeros((counts[k], len(active)), np.float32) for k in arrays}
        for col, i in enumerate(active):
            k, j = labels[i]
            selected[k][j, col] = mins[k][j]
    else:
        raise ValueError("Unknown grouping")
    config = dict(scan or {})
    config["motion"] = False
    if config.get("timing") is not None:
        config["timing"] = dict(config["timing"], velocity_voxels_s=[0., 0., 0.])
    baseline = scan_volume(
        neurons,
        optics,
        {k: v[:, None] for k, v in mins.items()},
        config,
        spikes=spikes,
        tpm=tpm,
        add_noise=False,
    )["clean"][:, :, 0]
    components = np.zeros(baseline.shape + (len(labels),), np.float32)
    if active:
        if config.get("timing") is not None:
            # Component columns are separate experiments, not a time series.
            # A timed scan must never interpolate one cell into another.
            rendered = np.concatenate([scan_volume(neurons, optics,
                {k: v[:, col:col+1] for k, v in selected.items()}, config,
                spikes=spikes, tpm=tpm, add_noise=False)["clean"]
                for col in range(len(active))], axis=2)
        else:
            rendered = scan_volume(
                neurons, optics, selected, config, spikes=spikes, tpm=tpm, add_noise=False
            )["clean"]
        components[:, :, active] = rendered
    return dict(
        components=components,
        baseline=baseline,
        ideal=ideal_support(components, baseline),
        labels=labels,
        definition="Separate component scans; overlaps need not sum to the combined baseline",
    )


def detector_snr(signal, background=0.0, noise=None):
    """Pre-rounding, pre-bleed expected detector SNR under the existing model."""
    p = dict(mu=100.0, sigma=2300.0, sigma0=2.7, darkcount=0.05)
    p.update(noise or {})
    if (
        not np.isfinite(list(p.values())).all()
        or p["mu"] <= 0
        or min(p["sigma"], p["sigma0"], p["darkcount"]) < 0
    ):
        raise ValueError("Invalid detector parameters")
    signal = np.asarray(signal, float)
    background = np.asarray(background, float)
    if (
        not np.isfinite(signal).all()
        or not np.isfinite(background).all()
        or np.any(signal < 0)
        or np.any(background < 0)
    ):
        raise ValueError("Nonnegative finite photon intensities required")
    variance = (p["mu"] ** 2 + p["sigma"]) * (signal + background + p["darkcount"]) + p[
        "sigma0"
    ] ** 2
    return p["mu"] * signal / np.sqrt(variance)
