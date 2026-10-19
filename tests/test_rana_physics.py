"""Independent analytic limits, direct sums and convergence checks for RANA."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import j0
from maomi import (FullFieldPSF, DepthPSFBank, DepthFullFieldRenderer, VectorialOptics,
                   StratifiedMedium, Acquisition, PulseSpectrum, EmissionResponse, RanaConfig, simulate)


class RanaPhysicsTests(unittest.TestCase):
    def test_matched_layer_and_paraxial_focus(self):
        m = StratifiedMedium(specimen_index=1.33)
        q = np.linspace(0, 1.1, 20)
        np.testing.assert_allclose(m.opd_um(q, 120), 0, atol=1e-12)
        m = StratifiedMedium(specimen_index=1.4, glass_um=180)
        # Paraxial refocusing removes q**2: leading nonzero term is q**4.
        ratio = m.opd_um(np.array([.02, .04]), 80)
        self.assertAlmostEqual(ratio[1]/ratio[0], 16, delta=.02)
        self.assertGreater(np.ptp(m.opd_um(q, 80)), .01)

    def test_scalar_debye_against_independent_bessel_integral(self):
        n, na, wl = 1.33, .7, .92
        medium = StratifiedMedium(n, n, n, n, 0, 0)
        optics = VectorialOptics(na=na, medium=medium, vectorial=False, radial_samples=30, azimuth_samples=80)
        x = np.linspace(-1, 1, 9)
        actual = optics.electric_field((x, [0], [0]), wavelength_nm=wl*1000)[:, 0, 0, 0]
        upper = np.arcsin(na/n)
        norm = quad(lambda t: np.sin(t)*np.sqrt(np.cos(t)), 0, upper, epsabs=1e-12)[0]
        expected = [quad(lambda t: np.sin(t)*np.sqrt(np.cos(t))*j0(2*np.pi*n/wl*xx*np.sin(t)),
                         0, upper, epsabs=1e-12)[0]/norm for xx in x]
        np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_vector_polarization_rotation_and_longitudinal_field(self):
        axes = (np.linspace(-.8, .8, 9), np.linspace(-.8, .8, 9), [0.])
        a = VectorialOptics(na=1.2, polarization=(1, 0)).electric_field(axes)
        b = VectorialOptics(na=1.2, polarization=(0, 1)).electric_field(axes)
        ia, ib = (np.sum(np.abs(f)**2, axis=-1) for f in (a, b))
        np.testing.assert_allclose(ia, ib.transpose(1, 0, 2), rtol=2e-12, atol=1e-13)
        self.assertGreater(np.sum(np.abs(a[..., 2])**2)/np.sum(np.abs(a)**2), .05)
        self.assertGreater(np.linalg.norm(ia-ib)/np.linalg.norm(ia), .1)

    def test_angular_convergence_and_depth_aberration(self):
        axes = (np.linspace(-.8, .8, 7), np.linspace(-.8, .8, 7), [-.4, 0, .4])
        a = VectorialOptics(na=1.1, radial_samples=18, azimuth_samples=40)
        b = VectorialOptics(na=1.1, radial_samples=36, azimuth_samples=80)
        p = a.response(axes, (0, 0, 100))
        ref = b.response(axes, (0, 0, 100))
        self.assertLess(np.linalg.norm(p-ref)/np.linalg.norm(ref), 2e-7)
        self.assertGreater(np.linalg.norm(p-a.response(axes, (0, 0, 0)))/np.linalg.norm(p), .01)

    def test_pulse_fluence_and_analytic_gaussian_chirp(self):
        for gdd in (0., 2000., -2000., 6000.):
            pulse = PulseSpectrum(gdd_fs2=gdd, samples=65)
            fields = np.ones((pulse.samples, 1), complex)
            self.assertAlmostEqual(float(pulse.integrate(fields, 1)), 1, places=12)
            expected = np.sqrt(2*np.log(2)/np.pi)/pulse.broadened_fwhm_fs
            self.assertAlmostEqual(float(pulse.integrate(fields, 2))/expected, 1, delta=2e-7)
            expected3 = (4*np.log(2)/np.pi)/(np.sqrt(3)*pulse.broadened_fwhm_fs**2)
            self.assertAlmostEqual(float(pulse.integrate(fields, 3))/expected3, 1, delta=2e-7)
        with self.assertRaisesRegex(ValueError, 'temporal window'):
            PulseSpectrum(gdd_fs2=10000, samples=19)

    def test_coherent_spectrum_and_incoherent_emission_are_distinct(self):
        pulse = PulseSpectrum(samples=33)
        e = np.ones((33, 3), complex)
        e[:, 1:] = 0
        coherent = pulse.integrate(e)
        e[:16] *= -1  # A spectral phase step splits the pulse (not a time translation).
        changed = pulse.integrate(e)
        self.assertGreater(abs(float(changed/coherent)-1), .1)
        response = EmissionResponse((500, 510, 530), (1, 1, 1), (.2, .4, .8))
        self.assertAlmostEqual(response.collection_fraction(), .5)

    def test_voxel_and_scan_quadrature_analytic_polynomial(self):
        class PolynomialOptics:
            def response(self, axes, position, **kwargs):
                x, y, z = np.meshgrid(*axes, indexing='ij')
                return (1+x*x+2*y*y+3*z*z)*(1+.1*position[0])
        # Uniform voxel variances pitch**2/12 and line displacement**2/12;
        # the odd pupil-gain cross term vanishes at the centered voxel.
        pitch = (.4, .6, .8)
        acq = Acquisition.line((.2, 0, 0), samples=3, voxel_order=2, dwell_seconds=.01)
        k = acq.kernel(PolynomialOptics(), (1, 1, 1), pitch, (0, 0, 10))
        expected = (1+.4**2/12+2*.6**2/12+3*.8**2/12+.2**2/12)*np.prod(pitch)*.01
        self.assertAlmostEqual(float(k.item()), expected, places=14)
        # Noncentral source: local scan-position gain must be sampled as well.
        k = acq.kernel(PolynomialOptics(), (3, 1, 1), pitch, (2, 0, 10))
        expected = (1.2*(1+.4**2+.4**2/12+2*.6**2/12+3*.8**2/12+.2**2/12)
                    - .2*.4*(.2**2/12))*np.prod(pitch)*.01
        self.assertAlmostEqual(k[2, 0, 0], expected, places=14)

    def test_depth_direct_sum_and_serialization(self):
        rng = np.random.default_rng(81)
        volume = rng.uniform(size=(4, 4, 3))
        kernels = rng.uniform(.1, 1, (2, 3, 3, 3))
        for operator in ('detection', 'excitation'):
            fields = tuple(FullFieldPSF(np.broadcast_to(k, (2, 2, 3, 3, 3)), [0, 4], [0, 4], (1, 1, 1), 920, operator)
                           for k in kernels)
            bank = DepthPSFBank([10, 13], fields)
            actual = DepthFullFieldRenderer(bank, volume.shape, voxel_size_um=(1, 1, 1), z_origin_um=10)(volume)
            direct = np.zeros_like(volume)
            for out in np.ndindex(volume.shape):
                for src in np.ndindex(volume.shape):
                    local = src if operator == 'detection' else out
                    offset = np.array(out)-src if operator == 'detection' else np.array(src)-out
                    ki = offset+1
                    if (ki < 0).any() or (ki >= 3).any():
                        continue
                    w = (local[2]+.5)/3
                    direct[out] += volume[src]*((1-w)*kernels[0][tuple(ki)]+w*kernels[1][tuple(ki)])
            np.testing.assert_allclose(actual, direct, rtol=2e-13, atol=2e-13)
            with tempfile.TemporaryDirectory() as tmp:
                bank.save(Path(tmp)/'bank.npz')
                restored = DepthPSFBank.load(Path(tmp)/'bank.npz')
                np.testing.assert_array_equal(restored.fields[1].kernels, fields[1].kernels)
            with self.assertRaisesRegex(ValueError, 'entire Z extent'):
                DepthFullFieldRenderer(bank, volume.shape, voxel_size_um=(1, 1, 1), z_origin_um=11)

    def test_config_runtime_and_cuda_equivalence(self):
        import torch
        cfg = RanaConfig((0, 2), (0, 2), (0, 2), (.5, .5, .5), kernel_shape=(3, 3, 3),
                         optics=VectorialOptics(radial_samples=4, azimuth_samples=8), acquisition=Acquisition(1))
        volume = np.ones((4, 4, 4))
        actual = simulate(volume, mode='rana', config=cfg, status_window=False)
        self.assertGreater(actual['timing_seconds']['optics'], 0)
        self.assertEqual(actual['settings']['depth_nodes_um'], [0., 2.])
        if torch.cuda.is_available():
            cuda = simulate(volume, mode='rana', config=cfg, device='cuda', status_window=False)
            np.testing.assert_allclose(actual['image'], cuda['image'], rtol=3e-12, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
