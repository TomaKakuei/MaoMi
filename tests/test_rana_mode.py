"""Explicit mode selection, UI isolation, packaging and CLI contracts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from maomi import FullFieldPSF, CubicFullFieldRenderer, simulate
from maomi.rana_ui import RanaStatus, ASSET, MESSAGE, sprite_frames
from maomi.__main__ import main


class RanaModeTests(unittest.TestCase):
    def setUp(self):
        self.a = np.arange(30.).reshape(5, 6)
        k = np.zeros((2, 2, 3, 3))
        k[..., 1, 1] = [[1, 2], [2, 3]]
        self.field = FullFieldPSF(k, [0, 5], [0, 6], (1, 1), 920, "excitation")

    def test_explicit_rana_preserves_local_operator(self):
        result = simulate(self.a, mode="rana", psf_field=self.field,
                          voxel_size_um=(1, 1), status_window=False)
        expected = CubicFullFieldRenderer(self.field, self.a.shape, voxel_size_um=(1, 1))(self.a)
        np.testing.assert_array_equal(result["image"], expected)
        self.assertEqual(result["mode"], "rana")
        self.assertEqual(result["status_window"], "disabled")
        self.assertFalse(result["settings"]["shared_psf_fallback"])
        self.assertAlmostEqual(result["simulation_seconds"], result["timing_seconds"]["optics"] + result["timing_seconds"]["preparation"]
                               + result["timing_seconds"]["render"])

    def test_standard_does_not_start_rana_window(self):
        with patch("maomi.fixed.run_fixed", return_value={"image": self.a}) as native, \
             patch("maomi.rana_ui.RanaStatus") as window:
            config = object()
            result = simulate(self.a, config=config, count_per_axis=32)
            native.assert_called_once_with(config, self.a, device="cuda", count_per_axis=32)
            window.assert_not_called()
            self.assertEqual(result["mode"], "standard")

    def test_mixed_or_missing_mode_inputs_rejected(self):
        for opts in [{"mode": "other"}, {"mode": "standard"},
                     {"mode": "standard", "config": object(), "psf_field": self.field},
                     {"mode": "rana"}, {"mode": "rana", "psf_field": self.field},
                     {"mode": "rana", "psf_field": self.field, "config": object()},
                     {"mode": "rana", "psf_field": self.field, "tails": False}]:
            with self.subTest(opts=opts), self.assertRaises((ValueError, TypeError)):
                simulate(self.a, status_window=False, **opts)

    def test_headless_path_is_numerically_identical(self):
        with patch("maomi.rana_ui.desktop_possible", return_value=False), \
             patch("maomi.rana_ui.subprocess.Popen") as start:
            shown = simulate(self.a, mode="rana", psf_field=self.field, voxel_size_um=(1, 1))
            start.assert_not_called()
        hidden = simulate(self.a, mode="rana", psf_field=self.field,
                          voxel_size_um=(1, 1), status_window=False)
        np.testing.assert_array_equal(shown["image"], hidden["image"])

    def test_unavailable_gui_does_not_cancel_computation(self):
        with patch("maomi.rana_ui.desktop_possible", return_value=True), \
             patch("maomi.rana_ui.subprocess.Popen", side_effect=OSError("no desktop")):
            out = simulate(self.a, mode="rana", psf_field=self.field, voxel_size_um=(1, 1))
        self.assertEqual(out["status_window"], "unavailable")
        self.assertTrue(np.isfinite(out["image"]).all())

    def test_window_cleanup_on_solver_error(self):
        with patch("maomi.rana_ui.RanaStatus") as status:
            with self.assertRaises(ValueError):
                simulate(np.ones((20, 20)), mode="rana", psf_field=self.field, voxel_size_um=(1, 1))
            self.assertIs(status.return_value.__exit__.call_args.args[0], ValueError)

    def test_cat_asset_and_rotation(self):
        with Image.open(ASSET) as img:
            self.assertEqual(img.mode, "RGBA")
            self.assertEqual(img.getextrema()[3], (0, 255))
        frames = sprite_frames(192)
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames[0].size, (192, 192))
        self.assertFalse(np.array_equal(np.asarray(frames[0]), np.asarray(frames[8])))
        self.assertEqual(MESSAGE, "MaoMi-RANA is working in background")



if __name__ == "__main__":
    unittest.main()
