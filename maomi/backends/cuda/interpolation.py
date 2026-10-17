"""Float64 MATLAB-v4 evaluation without a query-by-location matrix."""

from functools import lru_cache
from pathlib import Path
import numpy as np
import cupy as cp


@lru_cache(maxsize=8)
def _kernel(columns):
    source = Path(__file__).with_name("interpolation.cu").read_text()
    return cp.RawKernel(
        source, "green_apply", options=("--fmad=false", f"-DNVALUES={columns}")
    )


def evaluate_v4(points, coefficients, queries, *, chunk=16384):
    """Evaluate all value columns together; host inputs and output are float64.

    CUDA transfers and each launch are bounded by ``chunk`` queries. The
    returned NumPy output synchronizes completion. Compilation uses CuPy's
    normal cache (or the caller's CUPY_CACHE_DIR); no extra compiler is needed.
    """
    coefficients = np.asarray(coefficients, np.float64).reshape(len(points), -1)
    columns = coefficients.shape[1]
    points_gpu = cp.asarray(np.ascontiguousarray(points, dtype=np.float64))
    coeff_gpu = cp.asarray(np.ascontiguousarray(coefficients))
    output = np.empty((len(queries), columns), dtype=np.float64)
    kernel = _kernel(columns)
    for start in range(0, len(queries), chunk):
        q = cp.asarray(np.ascontiguousarray(queries[start : start + chunk], dtype=np.float64))
        value = cp.empty((len(q), columns), dtype=cp.float64)
        kernel(
            ((len(q) + 127) // 128,), (128,),
            (points_gpu, coeff_gpu, q, value, np.int32(len(points)), np.int32(len(q))),
        )
        value.get(out=output[start : start + len(q)])
    return output
