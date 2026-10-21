"""Acquisition integration, with asymmetric kernels and independent controls."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.signal import convolve
import torch

from maomi import RanaAddon, attach_rana
from maomi.enhanced import FullFieldPSF
from maomi.movie import single_scan, scan_movie, scan_volume, iter_scan_volume
from maomi.movie_io import write_movie
from maomi.scanning import zstack


def fixture(kernel_shape=(3, 4, 3)):
    rng = np.random.default_rng(631)
    shape = (12, 14, 9)
    kernel = rng.uniform(.1, 1, kernel_shape)
    kernel /= kernel.sum()
    points = np.array([[0, 0], [0, 14], [12, 0], [12, 14.]])
    optics = dict(psf=kernel, psf_3d=np.stack([kernel]*4), points_um=points,
                  mask=rng.uniform(.8, 1.2, shape[:2]),
                  colmask=rng.uniform(.7, 1, shape[:2]))
    for name in ('psfT', 'psfB'):
        optics[name] = dict(convmask=np.ones((3, 3)), weight=.04,
                            mask=rng.uniform(.8, 1.2, shape[:2]), psfZ=np.ones(9))
    optics.update(g_blur=True, blur=np.ones((3, 3))*.01)
    ids = rng.choice(np.prod(shape), 180, replace=False) + 1
    neurons = dict(neur_vol=np.zeros(shape),
                   gp_vals=[(ids[:60], rng.uniform(.2, 1, 60), np.arange(60) % 2 == 0),
                            (ids[60:120], rng.uniform(.2, 1, 60), np.arange(60) % 3 == 0)],
                   gp_nuc=[(ids[120:130], np.ones(10)), (ids[130:140], np.ones(10))],
                   bg_proc=[(ids[140:], rng.uniform(.2, .5, 40))])
    activity = dict(soma=np.array([[1., 1.7, 1.2], [.8, 1.4, 1.1]]),
                    dend=np.array([[1.2, 1., 1.4], [.7, 1.2, 1.]]),
                    bg=np.array([[.6, .8, .7]]))
    motion = [dict(x=3, y=np.full(shape[0], 3), z=z) for z in (2, 3, 4)]
    return optics, neurons, activity, motion


def addon(optics, field=None, device='cpu'):
    return attach_rana(optics, RanaAddon(field=field, voxel_size_um=(1.,)*3,
                                        device=device, status_window=False))


class AcquisitionAddon(unittest.TestCase):
    def test_disabled_identity_and_preserved_descriptors(self):
        optics, *_ = fixture()
        self.assertIs(attach_rana(optics, False), optics)
        enhanced = addon(optics)
        for key, value in optics.items():
            self.assertIs(enhanced[key], value)
        self.assertNotIn('rana', optics)
        self.assertFalse(enhanced['settings']['rana']['shared_psf_fallback'])

    def test_uniform_native_matches_static_and_grouped_scan(self):
        rng = np.random.default_rng(4)
        for ks in ((3, 4, 3), (4, 3, 4)):
            with self.subTest(kernel=ks):
                optics, *_ = fixture(ks)
                op = addon(optics)['rana']
                volume = rng.uniform(0, 1, (12, 14, 9))
                ref = zstack(volume, optics['psf'], mask=optics['mask'],
                             collection_mask=optics['colmask']).numpy()
                actual = zstack(volume, optics['psf'], mask=optics['mask'],
                                collection_mask=optics['colmask'], rana=op).numpy()
                np.testing.assert_allclose(actual, ref, atol=2e-14, rtol=1e-12)
                for group in (1, 2, 3):
                    slab = volume[..., :ks[2]]
                    np.testing.assert_allclose(single_scan(slab, optics['psf'], scan_avg=group, rana=op),
                                               single_scan(slab, optics['psf'], scan_avg=group),
                                               atol=2e-14, rtol=1e-12)

    def test_varying_local_shapes_reach_image_without_double_gain(self):
        optics, *_ = fixture()
        for n in range(4):
            optics['psf_3d'][n] = np.roll(optics['psf'], n % 3, axis=0) * (n+1)
        op = addon(optics)['rana']
        volume = np.random.default_rng(5).random((12, 14, 9))
        # Independent full convolutions with analytic 2x2 smoothstep weights.
        x = (np.arange(12)+.5)/12
        y = (np.arange(14)+.5)/14
        x, y = x*x*(3-2*x), y*y*(3-2*y)
        reference = np.zeros_like(volume)
        for i in range(2):
            for j in range(2):
                k = optics['psf_3d'][2*i+j]
                k = k/k.sum()*optics['psf'].sum()
                full = convolve(volume, k, mode='full', method='direct')
                crop = tuple(slice(s//2, s//2+n) for s, n in zip(k.shape, volume.shape))
                weight = (x if i else 1-x)[:, None]*(y if j else 1-y)[None, :]
                reference += full[crop]*weight[..., None]
        np.testing.assert_allclose(op.static(volume), reference, atol=1e-13)
        self.assertGreater(np.linalg.norm(reference-zstack(volume, optics['psf']).numpy()), .01)

    def test_sparse_movie_nucleus_tails_motion_binning_noise_and_multibeam(self):
        optics, neurons, activity, motion = fixture()
        enhanced = addon(optics)
        for nuclear in (0, 1):
            for sf in (1., 2., 1.5):
                for beams in (False, True):
                    with self.subTest(nuclear=nuclear, sfrac=sf, beams=beams):
                        a, b = optics, enhanced
                        if beams:
                            a = dict(channels=dict(a=optics, b=optics), beam_weights=dict(a=.3, b=1.4))
                            b = addon(a)
                        scan = dict(scan_buff=2, scan_avg=2, sfrac=sf, nuc_label=nuclear)
                        kw = dict(motion=motion, add_noise=False)
                        ref = scan_volume(neurons, a, activity, scan, **kw)
                        actual = scan_volume(neurons, b, activity, scan, **kw)
                        # The existing scanner uses float32 FFTs; assess absolute
                        # roundoff against the signal, including dark pixels.
                        error = actual['clean'].astype(float)-ref['clean']
                        self.assertLess(np.linalg.norm(error)/np.linalg.norm(ref['clean']), 2e-6)
                        self.assertLess(np.max(np.abs(error))/np.max(ref['clean']), 2e-6)
                        # Replay detector draws independently of tiny FFT rounding.
                        shape = ref['clean'].shape[:2]
                        draws = [dict(counts=np.full(shape, 3), gain_normal=np.zeros(shape),
                                      electronics_normal=np.ones(shape)) for _ in motion]
                        uniforms = [np.full(shape, .9) for _ in motion]
                        kw.update(add_noise=True, noise_draws=draws, bleed_uniforms=uniforms)
                        ref = scan_volume(neurons, a, activity, scan, **kw)
                        actual = scan_volume(neurons, b, activity, scan, **kw)
                        np.testing.assert_array_equal(actual['noisy'], ref['noisy'])
                        self.assertTrue(np.isfinite(actual['clean']).all())

    def test_dense_movie_and_streaming_export(self):
        optics, neurons, activity, motion = fixture()
        op = addon(optics)['rana']
        components = np.random.default_rng(6).random((2, 12, 14, 9))
        kw = dict(mask=optics['mask'], collection_mask=optics['colmask'],
                  z_start=[1, 2, 3], scan_avg=2, sfrac=2, buffer=2,
                  tails=[dict(kernel=np.ones((3, 3))*.01, indices=[0, 8], power_ratio=.2)])
        a = scan_movie(components, activity['soma'], optics['psf'], **kw)
        b = scan_movie(components, activity['soma'], optics['psf'], rana=op, **kw)
        np.testing.assert_allclose(a['clean'], b['clean'], rtol=2e-6)
        from maomi.movie import psf_fft
        kw['tails'] = [dict(kernel=psf_fft((12,14,9), np.ones((3,3))*.01),
                            indices=[0,8], power_ratio=.2, frequency=True)]
        a = scan_movie(components, activity['soma'], optics['psf'], **kw)
        b = scan_movie(components, activity['soma'], optics['psf'], rana=op, **kw)
        np.testing.assert_allclose(a['clean'], b['clean'], rtol=2e-6)
        scan = dict(scan_buff=2, sfrac=2)
        enhanced = addon(optics)
        ref = scan_volume(neurons, enhanced, activity, scan, motion=motion, add_noise=False)['clean']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'movie.npy'
            stream = iter_scan_volume(neurons, enhanced, activity, scan, motion=motion, add_noise=False)
            meta = write_movie(stream, path, frame_count=3, shape=ref.shape[:2])
            self.assertTrue(meta['complete'])
            np.testing.assert_array_equal(np.load(path), ref.transpose(2, 1, 0))

    def test_custom_depth_bank_changes_movie_and_reaches_analysis(self):
        from maomi.rana_depth import DepthPSFBank
        from maomi.analysis import ideal_components
        optics, neurons, activity, motion = fixture()
        fields = tuple(FullFieldPSF(np.stack([optics['psf']*gain]*4).reshape(2, 2, 3, 4, 3),
                                   [0, 12], [0, 14], (1.,)*3, 920., 'excitation')
                       for gain in (1., 3.))
        bank = DepthPSFBank([0., 10.], fields)
        enhanced = addon(optics, bank)
        scan = dict(scan_buff=2, sfrac=2)
        result = scan_volume(neurons, enhanced, activity, scan, motion=motion, add_noise=False)
        self.assertTrue(np.isfinite(result['clean']).all())
        self.assertGreater(np.linalg.norm(result['clean']-
                           scan_volume(neurons, optics, activity, scan, motion=motion, add_noise=False)['clean']), 1e-8)
        ideal = ideal_components(neurons, enhanced, activity, scan, grouping='by_compartment')
        self.assertTrue(np.isfinite(ideal['components']).all())

    def test_four_physics_provider_uses_existing_scanner(self):
        from maomi import RanaConfig, VectorialOptics, StratifiedMedium, Acquisition, PulseSpectrum, EmissionResponse
        optics, neurons, activity, motion = fixture()
        config = RanaConfig((0.,12.), (0.,14.), (0.,10.), (1.,)*3, kernel_shape=(3,4,3),
                           optics=VectorialOptics(na=1.1, medium=StratifiedMedium(specimen_index=1.38),
                                                 radial_samples=8, azimuth_samples=16),
                           acquisition=Acquisition.line((.1,0,0),samples=2,voxel_order=2),
                           pulse=PulseSpectrum(samples=33),
                           emission=EmissionResponse((500,525,550),(0,1,0),(.5,.8,.6)))
        enhanced = addon(optics, config)
        for nuclear in (0, 1):
            result = scan_volume(neurons, enhanced, activity,
                                 dict(scan_buff=2,sfrac=2,nuc_label=nuclear),
                                 motion=motion, rng=np.random.default_rng(81))
            self.assertTrue(np.isfinite(result['noisy']).all())
            self.assertGreater(result['clean'].sum(), 0)

    def test_custom_pitch_and_coverage_are_checked(self):
        optics, *_ = fixture()
        field = FullFieldPSF(np.ones((2,2,3,3,3)), [0,12], [0,14], (.5,)*3, 920., 'excitation')
        with self.assertRaisesRegex(ValueError, 'voxel size'):
            addon(optics, field)
        op = addon(optics)['rana']
        with self.assertRaisesRegex(ValueError, 'entire acquisition FOV'):
            op.static(np.ones((13,14,9)))

    def test_negative_tail_resampling_preserves_gain_and_noisy_movie(self):
        optics, neurons, activity, motion = fixture()
        op = addon(optics)['rana']
        raw = np.array([[1., -.05], [.3, .1]])
        fixed = op.physical_tail(raw)
        self.assertGreaterEqual(fixed.min(), 0)
        self.assertAlmostEqual(fixed.sum(), raw.sum())
        np.testing.assert_array_equal(raw, [[1., -.05], [.3, .1]])
        for key in ('psfT','psfB'):
            optics[key]['convmask'] = raw
        result = scan_volume(neurons, addon(optics), activity,
                             dict(scan_buff=2,sfrac=2,nuc_label=1), motion=motion,
                             rng=np.random.default_rng(35))
        self.assertTrue(np.isfinite(result['noisy']).all())

    def test_explicit_tail_policy_is_shared_and_preserves_native_descriptors(self):
        from copy import deepcopy
        from maomi.rana import RanaOperator
        optics, neurons, activity, motion = fixture()
        raw = np.array([[1., -.05], [.3, .1]], np.float32)
        for key in ('psfT', 'psfB'):
            optics[key]['convmask'] = raw.copy()
        physical = deepcopy(optics)
        for key in ('psfT', 'psfB'):
            physical[key]['convmask'] = RanaOperator.physical_tail(raw)
        for enabled in (False, True):
            original = addon(optics) if enabled else optics
            expected = addon(physical) if enabled else physical
            settings = dict(scan_buff=2, sfrac=2, nuc_label=1, tail_policy='nonnegative')
            actual = scan_volume(neurons, original, activity, settings, motion=motion,
                                 rng=np.random.default_rng(35))
            reference = scan_volume(neurons, expected, activity, dict(settings, tail_policy='native'),
                                    motion=motion, rng=np.random.default_rng(35))
            # A supplied descriptor is cast to float32 before normalization;
            # on-the-fly projection retains its float64 gain accumulator.
            np.testing.assert_allclose(actual['clean'], reference['clean'], rtol=1e-6, atol=1e-6)
            self.assertTrue(np.isfinite(actual['noisy']).all())
            np.testing.assert_array_equal(optics['psfT']['convmask'], raw)
            with self.assertRaisesRegex(ValueError, 'tail_policy'):
                scan_volume(neurons, original, activity, dict(settings, tail_policy='typo'), motion=motion)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_local_operator_matches_cpu(self):
        optics, *_ = fixture()
        for n in range(4):
            optics['psf_3d'][n] = np.roll(optics['psf'], n % 3, axis=0)
        volume = np.random.default_rng(7).random((12, 14, 9))
        cpu, gpu = addon(optics)['rana'], addon(optics, device='cuda')['rana']
        np.testing.assert_allclose(gpu.static(volume), cpu.static(volume), atol=2e-13)
        np.testing.assert_allclose(gpu.scan_slab(volume[..., :3], scan_avg=2),
                                   cpu.scan_slab(volume[..., :3], scan_avg=2), atol=2e-13)


if __name__ == '__main__':
    unittest.main()
