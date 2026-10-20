"""RANA local-kernel operator checks against explicit sums."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from maomi.enhanced import FullFieldPSF, CubicFullFieldRenderer
from maomi.psf import SampledPSF, PSFField


def field_for(shape, kernel_shape, operator):
    # Positive, asymmetric, affine-varying shapes AND gains: exact cubic control.
    rng = np.random.default_rng(18)
    base = rng.uniform(.2, 1, kernel_shape)
    dx = rng.uniform(0, .5, kernel_shape)
    dy = rng.uniform(0, .4, kernel_shape)
    def provider(x, y):
        return base + x / shape[0] * dx + y / shape[1] * dy
    field = FullFieldPSF.from_provider(provider, np.linspace(0, shape[0], 3),
                                      np.linspace(0, shape[1], 3),
                                      voxel_size_um=(1.,) * len(shape), wavelength_nm=920,
                                      operator=operator)
    return field, provider


def direct_sum(density, provider, kernel_shape, operator):
    out = np.zeros_like(density)
    center = np.array(kernel_shape) // 2
    for r in np.ndindex(density.shape):
        for s in np.ndindex(density.shape):
            node = s if operator == "detection" else r
            offset = np.array(r) - s if operator == "detection" else np.array(s) - r
            k = offset + center
            if np.all(k >= 0) and np.all(k < kernel_shape):
                out[r] += density[s] * provider(node[0] + .5, node[1] + .5)[tuple(k)]
    return out


class EnhancedContracts(unittest.TestCase):
    def test_affine_local_shapes_and_gain(self):
        f, provider = field_for((7, 5), (3, 4), "detection")
        renderer = CubicFullFieldRenderer(f, (7, 5), voxel_size_um=(1, 1))
        q = np.array([[0, 0], [7, 5], [.5, 4.5], [3.75, 2.25]])
        np.testing.assert_allclose(renderer.local_psfs(q), [provider(*p) for p in q], atol=2e-13)
        self.assertEqual(renderer.validation["checked_xy_positions"], 35)
        self.assertFalse(renderer.validation["shared_psf_fallback"])
        self.assertGreater(renderer.validation["minimum_interpolated_value"], 0)
        self.assertGreater(np.ptp(f.kernels.sum((2, 3))), 1)

    def test_direct_2d_3d_detection_and_excitation_even_odd(self):
        rng = np.random.default_rng(47)
        for shape in [(5, 4), (4, 3, 3)]:
            for kernel_shape in [(3,) * len(shape), (4, 2) + ((2,) if len(shape) == 3 else ())]:
                for operator in ["detection", "excitation"]:
                    with self.subTest(shape=shape, kernel_shape=kernel_shape, operator=operator):
                        f, provider = field_for(shape, kernel_shape, operator)
                        a = rng.uniform(0, 2, shape)
                        actual = CubicFullFieldRenderer(f, shape, voxel_size_um=(1,) * len(shape))(a)
                        expected = direct_sum(a, provider, kernel_shape, operator)
                        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def test_nonaffine_knots_and_between_nodes(self):
        # Independent cardinal-weight solve with genuinely nonlinear node values.
        from scipy.interpolate import RBFInterpolator
        f, _ = field_for((5, 4), (3, 3), "detection")
        kernels = f.kernels.copy()
        kernels[1, 1] *= 1.15
        f = FullFieldPSF(kernels, f.x_um, f.y_um, (1, 1), 920, "detection")
        renderer = CubicFullFieldRenderer(f, (5, 4), voxel_size_um=(1, 1))
        nodes = np.stack(np.meshgrid(f.x_um, f.y_um, indexing="ij"), -1).reshape(-1, 2)
        np.testing.assert_allclose(renderer.local_psfs(nodes), kernels.reshape(-1, 3, 3), atol=1e-12)
        ref = RBFInterpolator(nodes / [5, 4], kernels.reshape(9, -1), kernel="cubic", degree=1)
        provider = lambda x, y: ref(np.array([[x / 5, y / 4]]))[0].reshape(3, 3)
        a = np.random.default_rng(12).uniform(0, 1, (5, 4))
        np.testing.assert_allclose(renderer(a), direct_sum(a, provider, (3, 3), "detection"), atol=1e-12)

    def test_full_coverage_and_sampling_required(self):
        f, _ = field_for((5, 4), (3, 3), "excitation")
        for shape, pitch, origin in [((6, 4), (1, 1), (0, 0)),
                                     ((5, 4), (1, 1), (-.1, 0)),
                                     ((5, 4), (.5, 1), (0, 0))]:
            with self.subTest(shape=shape, pitch=pitch, origin=origin), self.assertRaises(ValueError):
                CubicFullFieldRenderer(f, shape, voxel_size_um=pitch, origin_um=origin)
        with self.assertRaises(TypeError):
            CubicFullFieldRenderer(SampledPSF.normalized(np.ones((3, 3)), (1, 1), 920),
                                   (5, 4), voxel_size_um=(1, 1))
        with self.assertRaises(ValueError):
            FullFieldPSF(f.kernels[:1], [0], f.y_um, (1, 1), 920, "excitation")
        renderer = CubicFullFieldRenderer(f, (5, 4), voxel_size_um=(1, 1))
        with self.assertRaises(ValueError):
            renderer.local_psfs([[5.01, 2]])

    def test_negative_cubic_overshoot_rejected(self):
        kernels = np.ones((3, 3, 2, 2)) * .01
        kernels[0, 0] = 100
        f = FullFieldPSF(kernels, [0, 4, 8], [0, 4, 8], (1, 1), 920, "detection")
        with self.assertRaisesRegex(ValueError, "nonphysical"):
            CubicFullFieldRenderer(f, (8, 8), voxel_size_um=(1, 1))

    def test_serialization_and_psf_field_adapter(self):
        original = PSFField.from_provider(lambda x, y: SampledPSF.normalized(
            np.ones((3, 3)), (.5, .5), 920), [0, 8], [0, 8])
        gains = np.array([[1, 2], [3, 4]])
        f = FullFieldPSF.from_psf_field(original, operator="excitation", gains=gains)
        np.testing.assert_allclose(f.kernels.sum((2, 3)), gains)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "field.npz"
            f.save(path)
            restored = FullFieldPSF.load(path)
            np.testing.assert_array_equal(restored.kernels, f.kernels)
            self.assertEqual(restored.operator, "excitation")
            self.assertEqual(restored.provenance, f.provenance)

    def test_nonzero_origin(self):
        f, _ = field_for((5, 4), (3, 3), "excitation")
        shifted = FullFieldPSF(f.kernels, f.x_um + 11, f.y_um - 3, (1, 1), 920, "excitation")
        a = np.ones((5, 4))
        expected = CubicFullFieldRenderer(f, a.shape, voxel_size_um=(1, 1))(a)
        actual = CubicFullFieldRenderer(shifted, a.shape, voxel_size_um=(1, 1), origin_um=(11, -3))(a)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA PyTorch required")
    def test_cuda_matches_direct_operator(self):
        for operator in ["detection", "excitation"]:
            f, provider = field_for((4, 3, 3), (4, 2, 3), operator)
            a = np.random.default_rng(18).uniform(0, 1, (4, 3, 3))
            actual = CubicFullFieldRenderer(f, a.shape, voxel_size_um=(1, 1, 1), device="cuda")(a)
            np.testing.assert_allclose(actual, direct_sum(a, provider, (4, 2, 3), operator), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
