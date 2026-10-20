import importlib.util
import unittest
import numpy as np
import torch
from scipy.signal import fftconvolve
from maomi.demo import synthetic_scene
from maomi.fixed import fixed_locations, CubicField, assemble, run_fixed
from maomi.optics import OpticalConfig, prepare_fields, propagate, locations, pupil

GPU = torch.cuda.is_available() and importlib.util.find_spec("cupy") is not None


class GridAndInterpolation(unittest.TestCase):
    def test_1024_unique_lattice_locations(self):
        for extent in [(16.0, 16.0, 10.0), (16.0, 24.0, 10.0)]:
            cfg = OpticalConfig(volume_um=extent, vres=2.0)
            p = fixed_locations(cfg, 32)
            self.assertEqual(p.shape, (1024, 2))
            self.assertEqual(len(np.unique(p, axis=0)), 1024)
            np.testing.assert_allclose(p * 4, np.round(p * 4))
            np.testing.assert_allclose(p.min(0), [0, 0])
            np.testing.assert_allclose(p.max(0), extent[:2])

    def test_main_defaults_to_4225_sites_and_rejects_shared_cubic(self):
        cfg, volume, _ = synthetic_scene()
        self.assertEqual(fixed_locations(cfg).shape, (4225, 2))
        with self.assertRaisesRegex(ValueError, "FullFieldRenderer"):
            run_fixed(cfg, volume, mask_mode="cubic")
        for count in [1, 3.5, True]:
            with self.assertRaises(ValueError):
                fixed_locations(cfg, count)

    def test_insufficient_lattice_rejected(self):
        with self.assertRaises(ValueError):
            fixed_locations(OpticalConfig(volume_um=(4.0, 4.0, 4.0)))

    def test_affine_fields_and_integral(self):
        cfg = OpticalConfig(volume_um=(16.0, 24.0, 10.0), vres=2.0)
        pts = locations(cfg.volume_um, 3)
        p = pts / np.array([16, 24])
        values = np.c_[2 + 3 * p[:, 0] - 4 * p[:, 1], 1 + p[:, 0] + p[:, 1]]
        weights, masks = CubicField(cfg, pts, (7, 5)).reconstruct(values)
        x, y = np.meshgrid((np.arange(7) + 0.5) / 7, (np.arange(5) + 0.5) / 5, indexing="ij")
        np.testing.assert_allclose(masks[..., 0], 2 + 3 * x - 4 * y, atol=1e-12)
        np.testing.assert_allclose(weights @ values, [1.5, 2.0], atol=1e-12)

    def test_phase_unit_conversion(self):
        cfg = OpticalConfig(volume_um=(16.0, 16.0, 10.0), depth_um=20.0, tail_um=10.0)
        phase = np.random.default_rng(92).normal(size=(17, 19)).astype(np.float32) * 0.1
        a = pupil(cfg, custom_phase=phase, phase_units="waves")
        b = pupil(cfg, custom_phase=phase * 2 * np.pi, phase_units="radians")
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=3e-7)


@unittest.skipUnless(GPU, "Requires CUDA PyTorch and CuPy")
class FullOpticalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)
        cls.cfg, cls.volume, cls.vessels = synthetic_scene("vessel", 20260927)
        cls.raw = prepare_fields(cls.cfg, cls.vessels)
        cls.full = run_fixed(cls.cfg, cls.volume, vessels=cls.vessels, raw_fields=cls.raw)

    def test_default_has_all_optical_descriptors(self):
        out = self.full
        self.assertEqual(out["unique_optical_paths"], 4225)
        self.assertEqual(out["mask_mode"], "matlab_v4")
        self.assertEqual(out["settings"]["count_per_axis"], 65)
        self.assertTrue(out["tails_computed"])
        for key in [
            "psf",
            "psf_raw",
            "psf_3d",
            "mask",
            "mask_raw",
            "colmask",
            "psfT",
            "psfB",
            "points_um",
            "image",
            "energies",
        ]:
            self.assertIn(key, out)
        for tail in ["psfT", "psfB"]:
            for name in [
                "convmask",
                "weight",
                "psfZ",
                "mask",
                "mask_raw",
                "weights_per_point",
                "profiles_per_point",
            ]:
                self.assertIn(name, out[tail])
                self.assertTrue(np.isfinite(out[tail][name]).all())
            self.assertGreater(out[tail]["weight"], 0)
            self.assertAlmostEqual(out[tail]["mask"].mean(), 1, places=12)

    def test_collection_mask_is_applied_to_final_image(self):
        out = self.full
        self.assertLess(out["colmask"].min(), 0.99)
        source = self.volume * out["mask"][..., None] * out["colmask"][..., None]
        p = out["psf"]
        full = fftconvolve(source, p, mode="full")
        crop = tuple(slice(k // 2, k // 2 + n) for k, n in zip(p.shape, source.shape))
        ref = full[crop]
        error = np.linalg.norm(out["image"] - ref) / np.linalg.norm(ref)
        self.assertLess(error, 2e-6)

    def test_focal_only_preserves_focal_image(self):
        out = run_fixed(
            self.cfg, self.volume, vessels=self.vessels, raw_fields=self.raw, tails=False
        )
        self.assertFalse(out["tails_computed"])
        self.assertNotIn("psfT", out)
        np.testing.assert_allclose(out["image"], self.full["image"], rtol=0, atol=0)

    def test_rana_reuses_native_excitation_scattering_phase_and_beams(self):
        from dataclasses import replace
        from maomi import RanaAddon
        phase = np.linspace(0., .4, 17*17).reshape(17,17)
        for mode in ("one-photon", "two-photon", "three-photon", "temporal-focusing"):
            with self.subTest(mode=mode):
                cfg = replace(self.cfg, scaling=mode)
                kw = dict(vessels=self.vessels, raw_fields=self.raw, count_per_axis=3,
                          custom_phase=phase, tails=True)
                standard = run_fixed(cfg, self.volume, **kw)
                local = run_fixed(cfg, self.volume, rana=RanaAddon(status_window=False), **kw)
                for key in ("psf_3d", "mask", "colmask", "psf"):
                    np.testing.assert_array_equal(standard[key], local[key])
                for key in ("psfT", "psfB"):
                    np.testing.assert_array_equal(standard[key]["convmask"], local[key]["convmask"])
                self.assertEqual(local["settings"]["scaling"], mode)
                self.assertTrue(np.isfinite(local["image"]).all())
                self.assertIn("rana", local["timing_seconds"])
        field = pupil(self.cfg).numpy()
        combined = run_fixed(self.cfg, self.volume, raw_fields=self.raw, vessels=self.vessels,
                             input_pupil={"left": field, "right": field},
                             beam_weights={"left": .7, "right": 1.3}, count_per_axis=3,
                             rana=RanaAddon(status_window=False))
        for channel in combined["channels"].values():
            self.assertIn("rana", channel)
        np.testing.assert_allclose(combined["image"],
                                   .7*combined["channels"]["left"]["image"]+
                                   1.3*combined["channels"]["right"]["image"])

    def test_assembled_tail_outputs_match_reference(self):
        from maomi.backends.cuda.batched import OpticalPlan

        cfg = OpticalConfig(volume_um=(16.0, 16.0, 10.0), depth_um=20.0, tail_um=10.0)
        raw = prepare_fields(cfg, scatter=False)
        points = locations(cfg.volume_um, 3)
        phase = np.zeros((17, 17), np.float32)
        phase[5:11, 5:11] = 0.2
        u = pupil(cfg, custom_phase=phase, device="cuda")
        plan = OpticalPlan(cfg, raw, uin=u)
        data = plan.evaluate(points, batch_size=4, tails=True)
        actual = assemble(
            cfg, data, (16, 16), mask_mode="matlab_v4", z_positions=plan.z_positions, cut=plan.cut
        )
        expected = propagate(cfg, raw, points, uin=u, device="cuda")
        for key in ["psf", "mask_raw", "psf_raw"]:
            self.assertLess(
                np.linalg.norm(actual[key] - expected[key]) / np.linalg.norm(expected[key]), 2e-5
            )
        for tail in ["psfT", "psfB"]:
            for key in ["convmask", "weight", "psfZ", "mask_raw"]:
                self.assertLess(
                    np.linalg.norm(np.asarray(actual[tail][key]) - expected[tail][key])
                    / np.linalg.norm(expected[tail][key]),
                    2e-5,
                )

    def test_invalid_volume_is_rejected(self):
        with self.assertRaises(ValueError):
            run_fixed(self.cfg, np.ones((3, 3, 3), np.float32))


if __name__ == "__main__":
    unittest.main()
