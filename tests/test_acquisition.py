"""Independent timing, physical limit, integration and workflow checks."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from maomi import AcquisitionClock, FullFieldPSF, RanaAddon, Scene, OpticalConfig, RunConfig, run_scene
from maomi.movie import scan_volume, detector_noise
from maomi._movie_buffers import ActivitySource


def fixture():
    shape = (12, 16, 3)
    x, y, z = np.indices(shape)
    density = np.exp(-((x-5.)**2+(y-7.)**2)/6)*(z == 1)
    ids = np.flatnonzero(density.ravel(order="F"))
    neurons = dict(neur_vol=density, gp_vals=[(ids+1, density.ravel(order="F")[ids], np.ones(len(ids), bool))], bg_proc=[])
    activity = dict(soma=np.array([[1., 2., 3.]]))
    psf = np.ones((1, 1, 1))
    optics = dict(psf=psf, mask=np.ones(shape[:2]), colmask=np.ones(shape[:2]))
    timing = dict(mode="rolling", exposure_s=.002, line_period_s=.001, samples=33,
                  photon_rate_scale=10000., velocity_voxels_s=[30., 0., 0.])
    scan = dict(scan_buff=2, scan_avg=1, sfrac=2, timing=timing)
    scene = Scene(neurons, OpticalConfig(volume_um=shape, psf_um=(1, 1, 1)), activity,
                  scan=scan, spikes=dict(dt=.04), noise=dict(bleedp=0))
    return scene, optics


class ExposureTests(unittest.TestCase):
    def test_constant_linear_rates_and_units(self):
        for mode in ("global", "rolling", "raster"):
            clock = AcquisitionClock(mode, .001, .01 if mode != "global" else 0,
                                     .001 if mode == "raster" else 0, samples=37)
            plan = clock.plan((7, 5))
            actual = plan.integrate(lambda t: np.full((7, 5), 20+3*t), center_s=.2)
            np.testing.assert_allclose(actual, .001*(20+3*(.2+plan.offsets)), rtol=1e-12)
            np.testing.assert_allclose(sum(plan.weights(t) for t in plan.nodes), .001, rtol=1e-12)

    def test_frame_midpoints_reverse_and_fast_scan(self):
        clock = AcquisitionClock("raster", .001, .01, .002, bidirectional=True)
        forward = clock.offsets((4, 3))
        reverse = replace(clock, reverse=True).offsets((4, 3))
        np.testing.assert_allclose(forward[:, ::-1], reverse)
        self.assertAlmostEqual(forward.mean(), 0.)
        self.assertGreater(forward[1, 0]-forward[0, 0], 0)
        self.assertLess(forward[1, 1]-forward[0, 1], 0)
        a = AcquisitionClock("global", .01).plan((9, 8))
        b = AcquisitionClock("rolling", .01, 0).plan((9, 8))
        f = lambda t: np.full((9, 8), 2+np.cos(t*20))
        np.testing.assert_array_equal(a.integrate(f), b.integrate(f))

    def test_motion_shape_and_temporal_refinement(self):
        # Analytic translating Gaussian: rolling timing induces affine shear.
        n = 129
        x, y = np.meshgrid(np.arange(n)-64., np.arange(n)-64., indexing="ij")
        sigma, velocity, line = 7., 800., .0005
        render = lambda t: np.exp(-((x-velocity*t)**2+y*y)/(2*sigma*sigma))
        clock = AcquisitionClock("rolling", .0003, line, samples=257)
        im = clock.plan((n, n)).integrate(render)
        fine = replace(clock, samples=513).plan((n, n)).integrate(render)
        self.assertLess(np.linalg.norm(im-fine)/np.linalg.norm(fine), .002)
        shear = np.sum(im*x*y)/np.sum(im*y*y)
        self.assertAlmostEqual(shear, velocity*line, places=5)

    def test_bad_clocks_and_rates(self):
        for kwargs in (dict(exposure_s=np.nan), dict(samples=1), dict(mode="globla"),
                       dict(line_period_s=.1), dict(mode="raster", pixel_period_s=0),
                       dict(mode="rolling", bidirectional=True)):
            with self.assertRaises(ValueError):
                AcquisitionClock(**kwargs)
        with self.assertRaises(ValueError):
            AcquisitionClock("raster", .001, .001, .001).plan((8, 8))
        for value in (-1., np.inf, complex(1, 2)):
            with self.assertRaises(ValueError):
                AcquisitionClock().plan((2, 2)).integrate(lambda t: np.full((2, 2), value))

    def test_activity_interpolation_across_chunks_and_boundaries(self):
        source = ActivitySource(dict(soma=np.array([[1., 3., 7., 9.]])), (1, 1, 0), chunk_frames=2)
        for t, expected in ((1.5, 5.), (-1, 1.), (4, 9.), (.5, 2.)):
            self.assertEqual(source.at_time(t)[0][0], expected)

    def test_timed_static_counts_and_collection_only_mask(self):
        scene, optics = fixture()
        scene.activity["soma"][:] = 1.
        scene.scan["timing"]["velocity_voxels_s"] = [0., 0., 0.]
        optics.pop("mask")
        optics["colmask"] *= .5
        out = run_scene(scene, settings=RunConfig(device="cpu", add_noise=False), optics=optics)
        density = scene.neurons["neur_vol"][2:-2, 2:-2, 1]
        expected = density.reshape(4, 2, 6, 2).sum(axis=(1, 3))*.5*10000*.002
        for t in range(3):
            np.testing.assert_allclose(out["clean"][..., t], expected, rtol=2e-6, atol=1e-6)

    def test_motion_precedes_spatial_psf(self):
        scene, optics = fixture()
        scene.activity["soma"][:] = 1.
        # Gain varies with detector X. Translating an already formed image
        # would move this gain field with the specimen, and fail this control.
        field = FullFieldPSF(np.array([1., 3.])[:, None, None, None, None]*np.ones((2, 2, 1, 1, 1)),
                             [0., 12.], [0., 16.], (1.,)*3, 920., operator="excitation")
        result = run_scene(scene, optics=optics, settings=RunConfig(device="cpu", add_noise=False),
                           rana=RanaAddon(field=field, status_window=False))
        from scipy.ndimage import shift
        sx = (np.arange(12)+.5)/12
        gain = 1+2*sx*sx*(3-2*sx)
        timing = dict(scene.scan["timing"])
        speed = timing.pop("velocity_voxels_s")
        rate_scale = timing.pop("photon_rate_scale")
        plan = AcquisitionClock(**timing).plan((4, 6))
        def reference(t):
            raw = shift(scene.neurons["neur_vol"], np.array(speed)*t, order=1, mode="constant", prefilter=False)[:, :, 1]
            rate = (raw*gain[:, None])[2:-2, 2:-2]
            return rate.reshape(4, 2, 6, 2).sum(axis=(1, 3))*rate_scale
        for t in range(3):
            np.testing.assert_allclose(result["clean"][..., t], plan.integrate(reference, center_s=t*.04), rtol=3e-6, atol=1e-6)

    def test_resume_and_multibeam_noise_once(self):
        scene, optics = fixture()
        field = FullFieldPSF(np.ones((2, 2, 1, 1, 1)), [0., 12.], [0., 16.], (1.,)*3, 920., "excitation")
        settings = RunConfig(device="cpu", seed=23, chunk_frames=1)
        for rana in (False, RanaAddon(field=field, status_window=False)):
            expected = run_scene(scene, settings=settings, optics=optics, rana=rana)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"movie"
                run_scene(scene, settings=settings, optics=optics, rana=rana, output=path, max_frames=1)
                finished = run_scene(output=path, resume=True)
                self.assertIn("acquisition_timing", finished.metadata)
                for t in range(3):
                    self.assertEqual(finished[t]["time_s"], t*.04)
                    for name in ("clean", "noisy"):
                        np.testing.assert_array_equal(finished[t][name], expected[name][..., t].T)
            channels = dict(channels=dict(a=optics, b=optics), beam_weights=dict(a=.5, b=.5))
            multibeam = run_scene(scene, settings=settings, optics=channels, rana=rana)
            for name in ("clean", "noisy"):
                np.testing.assert_array_equal(multibeam[name], expected[name])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA optional")
    def test_rana_cpu_cuda_timed(self):
        scene, optics = fixture()
        field = FullFieldPSF(np.ones((2, 2, 1, 1, 1)), [0., 12.], [0., 16.], (1.,)*3, 920., "excitation")
        results = [run_scene(scene, optics=optics, settings=RunConfig(device="cpu", add_noise=False),
                    rana=RanaAddon(field=field, device=device, status_window=False))["clean"] for device in ("cpu", "cuda")]
        np.testing.assert_allclose(*results, rtol=5e-5, atol=2e-6)

    def test_invalid_detector_parameters(self):
        for key in ("mu", "mu0", "sigma", "sigma0", "darkcount"):
            with self.assertRaisesRegex(ValueError, "finite"):
                detector_noise(np.ones((2, 2)), {key: np.nan})

    def test_static_cache_avoids_repeated_optical_calls(self):
        from maomi.movie import single_scan
        scene, optics = fixture()
        scene.activity["soma"][:] = 1.
        scene.scan["timing"]["velocity_voxels_s"] = [0., 0., 0.]
        with patch("maomi.movie.single_scan", wraps=single_scan) as scanner:
            run_scene(scene, optics=optics, settings=RunConfig(device="cpu", add_noise=False))
        self.assertEqual(scanner.call_count, 1)

    def test_psf_exposure_is_not_counted_twice(self):
        scene, optics = fixture()
        field = FullFieldPSF(np.ones((2, 2, 1, 1, 1)), [0., 12.], [0., 16.], (1.,)*3, 920., "excitation")
        settings = RunConfig(device="cpu", add_noise=False)
        reference = run_scene(scene, settings=settings, optics=optics,
                              rana=RanaAddon(field=field, status_window=False))["clean"]
        for provenance in (dict(kernel_exposure_s=.004),
                           dict(model="RANA Richards-Wolf layered pupil", acquisition=dict(dwell_seconds=.004))):
            integrated = replace(field, kernels=field.kernels*.004, provenance=provenance)
            result = run_scene(scene, settings=settings, optics=optics,
                               rana=RanaAddon(field=integrated, status_window=False))["clean"]
            np.testing.assert_allclose(result, reference, rtol=2e-6, atol=1e-6)
            np.testing.assert_array_equal(integrated.kernels, field.kernels*.004)

    def test_tails_nuclear_and_ideal_components(self):
        from test_rana_addon import fixture as detailed_fixture, addon
        from maomi.analysis import ideal_components
        optics, neurons, activity, _ = detailed_fixture()
        scan = dict(scan_buff=2, sfrac=2, scan_avg=1,
                    timing=dict(mode="rolling", exposure_s=.001, line_period_s=.001,
                                samples=5, velocity_voxels_s=[1, 0, 0]))
        for selected in (optics, addon(optics)):
            for nuclear in (0, 1):
                result = scan_volume(neurons, selected, activity, dict(scan, nuc_label=nuclear), add_noise=False)
                self.assertTrue(np.isfinite(result["clean"]).all())
                self.assertGreater(result["clean"].max(), 0)
        actual = ideal_components(neurons, optics, activity, scan, grouping="by_compartment")
        mins = {k: np.asarray(v).min(axis=1)[:, None] for k, v in activity.items()}
        stationary = dict(scan, timing=dict(scan["timing"], velocity_voxels_s=[0, 0, 0]))
        for column, (key, index) in enumerate(actual["labels"]):
            one = {k: np.zeros_like(v) for k, v in mins.items()}
            one[key][index] = mins[key][index]
            expected = scan_volume(neurons, optics, one, stationary, add_noise=False)["clean"][..., 0]
            np.testing.assert_array_equal(actual["components"][..., column], expected)


if __name__ == "__main__":
    unittest.main()
