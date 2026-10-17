"""Independent numerical contracts for the additive platform prototype."""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
from maomi.integrations import SampledPSF, pupil_to_intensity
from maomi.synthetic import procedural_cells, expected_measurement, noisy_measurement


class PlatformTests(unittest.TestCase):
    def test_zero_opd_piston_and_fft_origin(self):
        # Zero and integer-wave OPDs transmit; a constant piston cannot change intensity.
        for n in (16, 17):
            opd = np.zeros((n, n)); opd[0, :] = np.nan
            p = pupil_to_intensity(opd, 64)
            q = pupil_to_intensity(opd+1.234, 64)
            np.testing.assert_allclose(p, q, atol=1e-15)
            self.assertEqual(np.unravel_index(p.argmax(), p.shape), (32, 32))
            self.assertAlmostEqual(float(p.sum()), 1.)

    def test_tilt_uses_x_axis(self):
        n = 32
        opd = np.broadcast_to(3*np.arange(n)[:, None]/n, (n, n))
        p = pupil_to_intensity(opd, n)
        self.assertEqual(np.unravel_index(p.argmax(), p.shape), (19, 16))
        self.assertGreater(p[19, 16], 1.-1e-12)

    def test_psf_roundtrip_and_validation(self):
        p = SampledPSF.normalized(np.ones((5, 4)), (.2, .3), 550., {'provider':'test'})
        with tempfile.TemporaryDirectory() as tmp:
            p.save(Path(tmp)/'psf.npz')
            q = SampledPSF.load(Path(tmp)/'psf.npz')
            np.testing.assert_array_equal(p.intensity, q.intensity)
            self.assertEqual(p.provenance, q.provenance)
        for bad in [np.zeros((4,4)), -np.ones((4,4)), np.full((4,4), np.nan)]:
            with self.assertRaises(ValueError):
                SampledPSF.normalized(bad, (1., 1.), 550.)

    def test_impulse_convolution_even_and_odd(self):
        # Asymmetric kernels expose orientation/cropping errors that symmetric PSFs hide.
        for n in (4, 5):
            raw = np.arange(1., n*n+1).reshape(n,n)
            p = SampledPSF.normalized(raw, (.4,.4), 550.)
            d = np.zeros((15,15)); d[7,7] = 1.
            actual = expected_measurement(d, p, pixel_size_um=(.4,.4), photons_per_unit=2., background_e=3.)
            reference = np.full_like(d, 3.)
            start = 7-n//2
            reference[start:start+n,start:start+n] += 2*p.intensity
            np.testing.assert_allclose(actual, reference, atol=1e-12)
        with self.assertRaises(ValueError):
            expected_measurement(d, p, pixel_size_um=(1.,1.))

    def test_procedural_additivity(self):
        cell = dict(center_um=[10.,10.], radii_um=[3.,2.], brightness=1., nucleus_fraction=.5)
        one, _ = procedural_cells((32,32), (.6,.6), [cell])
        two, layers = procedural_cells((32,32), (.6,.6), [cell, cell])
        np.testing.assert_array_equal(two, 2*one)
        np.testing.assert_array_equal(layers.sum(axis=0), two)
        self.assertGreater(one.sum(), 0.)

    def test_detector_moments_and_unclipped_noise(self):
        # Mean/variance have independent Poisson-Gaussian analytic references.
        observed = noisy_measurement(np.full(200000, 7.), rng=np.random.default_rng(1024), read_sigma_e=2.)
        self.assertLess(abs(observed.mean()-7.), .04)
        self.assertLess(abs(observed.var()-11.), .15)
        dark = noisy_measurement(np.zeros(20000), rng=np.random.default_rng(22), read_sigma_e=2.)
        self.assertTrue((dark < 0).any())
        self.assertLess(abs(dark.mean()), .05)

    @unittest.skipUnless(importlib.util.find_spec('rayoptics'), 'optional RayOptics not installed')
    def test_real_rayoptics_in_fresh_interpreter(self):
        # Other tests must not hide first-use circular imports by preloading modules.
        code = (
            "from maomi.integrations.rayoptics import demo_lens, rayoptics_psf\n"
            "psf = rayoptics_psf(demo_lens(), num_rays=17, fft_size=128)\n"
            "assert psf.intensity.shape == (128, 128)\n"
            "assert abs(psf.intensity.sum() - 1.) < 1e-12\n"
        )
        result = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True,
            timeout=60, env=dict(os.environ, MPLBACKEND='Agg'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('rayoptics'), 'optional RayOptics not installed')
    def test_real_rayoptics_sampling_and_lens_change(self):
        from maomi.integrations.rayoptics import demo_lens, rayoptics_psf
        p = rayoptics_psf(demo_lens(), num_rays=17, fft_size=128)
        q = rayoptics_psf(demo_lens(), num_rays=17, fft_size=256)
        np.testing.assert_allclose(p.pixel_size_um, 2*np.asarray(q.pixel_size_um), rtol=1e-12)
        # Independent thin-lens pupil scaling: lambda / (N * delta_u * object-side NA).
        expected_um = .55/(128*(2.04/16)*.04)
        self.assertAlmostEqual(p.pixel_size_um[0], expected_um, places=10)
        defocused = rayoptics_psf(demo_lens(), num_rays=17, fft_size=128, focus_offset=.5)
        self.assertGreater(np.linalg.norm(p.intensity-defocused.intensity)/np.linalg.norm(p.intensity), .01)


if __name__ == '__main__':
    unittest.main()
