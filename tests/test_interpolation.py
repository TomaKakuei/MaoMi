import unittest
from importlib.util import find_spec
import numpy as np
import torch
from maomi.numerics import v4_interpolate


def original_v4(points, values, queries):
    """Frozen pre-optimization equation, independent of the new evaluators."""
    def green(a, b):
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            g = d * d * (np.log(d) - 1)
        g[d == 0] = 0
        return g
    c = np.linalg.solve(green(points, points), values)
    flat = np.asarray(queries).reshape(-1, 2)
    return (green(flat, points) @ c).reshape(queries.shape[:-1] + values.shape[1:])


class Interpolation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(471)
        self.points = rng.uniform(-20, 20, (37, 2))
        self.queries = np.concatenate((self.points, rng.uniform(-25, 25, (105, 2))))
        self.values = rng.uniform(.2, 2, (37, 4))

    def check_backend(self, backend):
        for values in (self.values[:, 0], self.values[:, :3], self.values):
            for queries in (self.queries, self.queries.reshape(2, 71, 2)):
                actual = v4_interpolate(self.points, values, queries, chunk=19, backend=backend)
                ref = original_v4(self.points, values, queries)
                np.testing.assert_allclose(actual, ref, rtol=1e-9, atol=1e-10)
                self.assertEqual(actual.dtype, np.dtype("float64"))

    def test_cpu_matches_frozen_equation_including_nodes_and_extrapolation(self):
        self.check_backend("cpu")

    @unittest.skipUnless(torch.cuda.is_available() and find_spec("cupy"), "CUDA/CuPy required")
    def test_cuda_matches_frozen_equation_for_multiple_columns_and_batches(self):
        self.check_backend("cuda")

    def test_single_point_and_empty_queries(self):
        actual = v4_interpolate([[3, 4]], [[2, 7]], np.zeros((5, 6, 2)))
        np.testing.assert_array_equal(actual, np.broadcast_to([2, 7], (5, 6, 2)))
        empty = v4_interpolate(self.points, self.values, np.empty((0, 2)))
        self.assertEqual(empty.shape, (0, 4))

    def test_invalid_backend_and_batch_size(self):
        for kwargs in ({"backend": "unknown"}, {"chunk": 0}, {"chunk": -1}):
            with self.assertRaises(ValueError):
                v4_interpolate(self.points, self.values, self.queries, **kwargs)


if __name__ == "__main__":
    unittest.main()
