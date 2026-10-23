"""Image comparison in original units, with explicitly scoped ROI metrics."""
import numpy as np


def compare_images(reference, candidate, *, roi=None):
    """Summarize a finite real pair without alignment or per-image scaling.

    ROI is ((x0,x1),(y0,y1)), zero-based and half-open. A zero reference norm
    gives relative_l2=None for a nonzero candidate, zero for identical zeros.
    Values are fractions, not percentages. Signed residual = candidate-reference.
    """
    ref, other = np.asarray(reference), np.asarray(candidate)
    if (ref.ndim != 2 or ref.shape != other.shape or np.iscomplexobj(ref)
            or np.iscomplexobj(other) or not np.isfinite(ref).all() or not np.isfinite(other).all()):
        raise ValueError("Comparison requires finite real XY arrays of identical shape")
    ref, other = ref.astype(float), other.astype(float)
    def measure(a, b):
        difference = b-a
        norm, error = float(np.linalg.norm(a)), float(np.linalg.norm(difference))
        return dict(relative_l2=error/norm if norm else (0. if error == 0 else None),
                    max_abs=float(np.max(np.abs(difference))), mean_signed=float(difference.mean()),
                    identical=bool(np.array_equal(a, b)), pixels=a.size)
    result = dict(full_field=measure(ref, other),
                  display_range=[float(min(ref.min(), other.min())), float(max(ref.max(), other.max()))],
                  residual="candidate - reference; original units; no fitted alignment or gain")
    if roi is not None:
        if len(roi) != 2 or any(len(pair) != 2 for pair in roi):
            raise ValueError("ROI must be ((x0,x1),(y0,y1))")
        for pair, n in zip(roi, ref.shape):
            lo, hi = pair
            if int(lo) != lo or int(hi) != hi or not 0 <= lo < hi <= n:
                raise ValueError("ROI bounds must lie inside the image")
        region = tuple(slice(int(lo), int(hi)) for lo, hi in roi)
        result["roi"] = dict(bounds=[list(p) for p in roi], **measure(ref[region], other[region]))
    return result
