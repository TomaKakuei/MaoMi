import importlib.util
import unittest
import numpy as np
from maomi import OpticalConfig, generate_scene, tissue_parameters


class TissueGridTests(unittest.TestCase):
    def test_aligned_defaults_and_explicit_blocks_are_retained(self):
        cfg = OpticalConfig(volume_um=(100, 100, 60))
        params = tissue_parameters(cfg, preset="naomi")
        np.testing.assert_array_equal(params["dend"]["dimsSS"], [5, 5, 5])
        params = tissue_parameters(
            cfg, preset="naomi", overrides={"dend": {"dimsSS": [2, 4, 3]}}
        )
        np.testing.assert_array_equal(params["dend"]["dimsSS"], [2, 4, 3])

    @unittest.skipUnless(importlib.util.find_spec("numba"), "Install the tissue extra")
    def test_background_integer_parameters_match_float_at_fractional_resolution(self):
        from maomi._tissue_growth import generate_background
        from maomi._tissue_random import RandomStream
        cfg = OpticalConfig(volume_um=(12, 12, 8), vres=1.5)
        params = tissue_parameters(cfg, preset="naomi", overrides={
            "vol": {"N_neur": 1, "N_den": 0, "N_bg": 0},
            "dend": {"dtParams": [2, 4, 3, 1, 0]},
        })
        shape = (18, 18, 12)
        volume = np.zeros(shape, np.float32)
        labels = np.zeros(shape, np.uint16)
        components = [[np.array([100]), np.array([1.]), np.array([True])]]
        outputs = []
        for dtype in (int, float):
            dend = dict(params["dend"], dtParams=np.array([2, 4, 3, 1, 0], dtype=dtype))
            outputs.append(generate_background(
                params["vol"], params["bg"], dend, volume, labels,
                components, [], np.array([[4., 4., 4.]]),
                random_stream=RandomStream(seed=39),
            ))
        for i in (0, 1, 4):
            np.testing.assert_array_equal(outputs[0][i], outputs[1][i])
        self.assertEqual(len(outputs[0][3]), len(outputs[1][3]))
        for a, b in zip(outputs[0][3], outputs[1][3]):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    @unittest.skipUnless(importlib.util.find_spec("numba"), "Install the tissue extra")
    def test_nonmultiple_tissue_grows_main_and_apical_dendrites(self):
        cfg = OpticalConfig(
            volume_um=(32, 32, 24), depth_um=10, psf_um=(4, 4, 4), tail_um=5
        )
        scene = generate_scene(cfg, preset="naomi", seed=731, overrides={
            "vol": {"N_neur": 1, "N_den": 1, "N_bg": 0},
            "vasc": {"flag": 0}, "bg": {"flag": 0}, "axon": {"flag": 0},
            "dend": {"dtParams": [3, 8, 4, 1, 0]},
        })
        self.assertEqual(scene.shape, (32, 32, 24))
        self.assertEqual(scene.config, cfg)
        self.assertTrue(np.isfinite(scene.neurons["neur_vol"]).all())
        self.assertGreater(np.count_nonzero(scene.neurons["neur_vol"]), 0)
        stages = scene.metadata["tissue"]["timing_seconds"]
        self.assertIn("growNeuronDendrites", stages)
        self.assertIn("growApicalDendrites", stages)


if __name__ == "__main__":
    unittest.main()
