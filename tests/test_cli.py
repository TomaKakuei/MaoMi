"""Common CLI contracts, real acquisition and RANA package independence."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from maomi.cli import main, build_parser
from maomi import MovieDataset, load_optics, load_scene, render_scene, RunConfig
from maomi.rana import FullFieldPSF, RanaConfig


def invoke(args):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), patch.dict(os.environ, {"MAOMI_STATUS_WINDOW": "0"}):
        main([str(arg) for arg in args])
    return json.loads(stream.getvalue())


class CommonCLI(unittest.TestCase):
    def test_help_and_entrypoints_share_one_parser(self):
        from maomi.__main__ import main as module_main
        from maomi.workflow_cli import main as compatibility_main
        self.assertIs(module_main, main)
        self.assertIs(compatibility_main, main)
        help_text = build_parser().format_help()
        for word in ("run", "render", "resume", "inspect", "--rana"):
            self.assertIn(word, help_text)
        for cmd in ("run", "render"):
            a = build_parser().parse_args([cmd, "--demo", "--rana", "--device", "cpu", "--output", "unused"])
            self.assertTrue(a.rana)

    def test_removed_mode_and_inactive_field_fail_before_work(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"unused"
            for args in (["--mode", "rana"], ["render", "--mode=rana"],
                         ["run", "--demo", "--psf-field", "missing.npz", "--output", output]):
                with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as exc:
                    invoke(args)
                self.assertEqual(exc.exception.code, 2)
                self.assertIn("--rana", stderr.getvalue())
                self.assertFalse(output.exists())

    def test_standard_and_rana_movies_resume_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rana in (False, True):
                args = ["run", "--demo", "--device", "cpu", "--frames", "5", "--seed", "37"]
                if rana:
                    args.append("--rana")
                full, partial = root/f"full_{rana}", root/f"partial_{rana}"
                invoke(args+["--output", full])
                self.assertFalse(invoke(args+["--output", partial, "--max-frames", "2"])["complete"])
                self.assertTrue(invoke(["resume", partial])["complete"])
                a, b = MovieDataset(full, verify=True), MovieDataset(partial, verify=True)
                for frame in range(5):
                    for key in ("clean", "noisy"):
                        np.testing.assert_array_equal(a[frame][key], b[frame][key])
                meta = invoke(["inspect", partial])
                self.assertEqual(meta["settings"]["rana"], rana)
                self.assertEqual("rana" in load_optics(partial/"optics.h5"), rana)

    def test_render_uses_same_workflow_and_directory_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rana in (False, True):
                output = root/str(rana)
                args = ["render", "--demo", "--device", "cpu", "--output", output]
                if rana:
                    args.append("--rana")
                invoke(args)
                optics = load_optics(output/"optics.h5")
                scene = load_scene(output/"scene.h5")
                expected = render_scene(scene, optics=optics, settings=RunConfig(device="cpu"))["image"]
                with np.load(output/"image.npz") as arrays:
                    np.testing.assert_array_equal(arrays["image"], expected)
                    self.assertGreater(float(expected.sum()), 0)
                self.assertEqual("rana" in optics, rana)
                self.assertEqual(invoke(["inspect", output])["schema"], "maomi.render.v1")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    invoke(args)

    def test_custom_field_reaches_both_commands_and_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kernels = np.zeros((2, 2, 3, 3, 3))
            kernels[..., 1, 1, 1] = [[1, 2], [3, 4]]
            field = FullFieldPSF(kernels, [0, 16], [0, 16], (1, 1, 1), 920, "excitation")
            field.save(root/"field.npz")
            for command in ("render", "run"):
                result = invoke([command, "--demo", "--device", "cpu", "--rana",
                                 "--psf-field", root/"field.npz", "--output", root/command])
                saved = load_optics(root/command/"optics.h5")["rana"]
                np.testing.assert_array_equal(saved.field.kernels, kernels)
                self.assertFalse(saved.native)
                self.assertFalse(saved.settings["shared_psf_fallback"])
                if command == "run":
                    self.assertTrue(result["complete"])
            self.assertEqual(invoke(["inspect", root/"field.npz"])["kind"], "FullFieldPSF")
            # A partial field cannot silently fall back to a mean PSF.
            FullFieldPSF(kernels, [0, 8], [0, 8], (1, 1, 1), 920, "excitation").save(root/"short.npz")
            with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
                invoke(["render", "--demo", "--device", "cpu", "--rana", "--psf-field",
                        root/"short.npz", "--output", root/"invalid"])
            self.assertIn("entire acquisition FOV", stderr.getvalue())
            self.assertFalse((root/"invalid").exists())

    def test_core_renders_without_importing_host_or_torch(self):
        code = '''
import sys
import numpy as np
from maomi.rana import FullFieldPSF, RanaOperator
k = np.zeros((2,2,3,3,3)); k[...,1,1,1] = 1
field = FullFieldPSF(k, [0,4], [0,4], (1,1,1), 920, 'excitation')
actual = RanaOperator(field).static(np.ones((4,4,3)))
np.testing.assert_allclose(actual, 1, atol=1e-12)
for name in ('maomi.optics','maomi.fixed','maomi.movie','maomi.tissue_pipeline','torch','cupy'):
    assert name not in sys.modules, name
'''
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_volume_and_relative_field_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            volume = np.ones((8, 8, 6))
            np.save(root/"density.npy", volume)
            k = np.zeros((2, 2, 3, 3, 3)); k[..., 1, 1, 1] = 1
            FullFieldPSF(k, [0, 8], [0, 8], (1, 1, 1), 920, "excitation").save(root/"field.npz")
            cfg = dict(optical=dict(volume_um=[8, 8, 6], depth_um=10, psf_um=[4, 4, 4], tail_um=5),
                       run=dict(device="cpu", rana=True, count_per_axis=2), rana=dict(field="field.npz"))
            (root/"config.json").write_text(json.dumps(cfg), encoding="utf8")
            invoke(["render", "--volume", root/"density.npy", "--config", root/"config.json", "--output", root/"out"])
            with np.load(root/"out/image.npz") as arrays:
                np.testing.assert_array_equal(arrays["volume"], volume)
                self.assertTrue(np.isfinite(arrays["image"]).all())
            self.assertEqual(invoke(["inspect", root/"out/scene.h5"])["shape"], list(volume.shape))

    def test_physics_json_is_same_configuration_as_python(self):
        specification = dict(x_um=[0, 4], y_um=[0, 4], depths_um=[10, 12], voxel_size_um=[1, 1, 1],
                             kernel_shape=[3, 3, 3], optics=dict(radial_samples=4, azimuth_samples=8),
                             acquisition=dict(voxel_order=1))
        cfg = RanaConfig.from_dict(specification)
        from maomi.rana import VectorialOptics, Acquisition
        expected = RanaConfig((0, 4), (0, 4), (10, 12), (1, 1, 1), kernel_shape=(3, 3, 3),
                              optics=VectorialOptics(radial_samples=4, azimuth_samples=8),
                              acquisition=Acquisition(voxel_order=1))
        for a, b in zip(cfg.build_field().fields, expected.build_field().fields):
            np.testing.assert_array_equal(a.kernels, b.kernels)


if __name__ == "__main__":
    unittest.main()
