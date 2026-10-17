import unittest
import numpy as np
import torch
from scipy.signal import convolve
from maomi.scanning import zstack
from maomi.optics import OpticalConfig, locations, run_optics


class Contracts(unittest.TestCase):
    def test_even_kernel_matches_full_convolution_crop(self):
        v = np.zeros((7, 8, 9), np.float64)
        v[1, 2, 3] = 1
        p = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        full = convolve(v, p, mode="full", method="direct")
        ref = full[1:8, 1:9, 2:11]
        np.testing.assert_allclose(zstack(v, p).numpy(), ref, atol=1e-13)

    def test_center_and_physical_energy_contract(self):
        cfg = OpticalConfig(volume_um=(16.0, 16.0, 10.0), depth_um=20.0, tail_um=10.0)
        pts = locations(cfg.volume_um, 1)
        np.testing.assert_array_equal(pts, [[8, 8]])
        out = run_optics(cfg, pts, scatter=False)
        self.assertGreater(out["psf"].sum(), 0)
        self.assertGreater(out["psfT"]["weight"], 0)
        self.assertGreater(out["psfB"]["weight"], 0)
        np.testing.assert_allclose(out["mask"], 1)
        np.testing.assert_allclose(out["colmask"], 1)
        # Upstream averages abs(PSFs), but its mask sums signed cubic samples.
        # The raw output preserves that distinction rather than silently fixing it.
        self.assertAlmostEqual(float(out["psf_raw"].sum()), out["mask_mean"], places=5)
        np.testing.assert_allclose(out["psf"], np.abs(out["psf_3d"]).mean(0))


if __name__ == "__main__":
    unittest.main()
