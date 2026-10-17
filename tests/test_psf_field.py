"""Independent source-by-source checks of the spatially varying operator."""
import tempfile
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from maomi.psf import SampledPSF, PSFField, resample_psf
from maomi.imaging import PSFRenderer, render_image
from maomi.synthetic import expected_measurement
from maomi.scanning import zstack


def direct_reference(density, field, origin=(0., 0.)):
    """Scatter each source's interpolated kernel directly, without convolution/FFTs."""
    lookup = RegularGridInterpolator((field.x_um, field.y_um), field.intensity)
    kx, ky = field.intensity.shape[-2:]
    full = np.zeros((density.shape[0]+kx-1, density.shape[1]+ky-1))
    for i, j in np.ndindex(density.shape):
        point = [np.clip(origin[0]+(i+.5)*field.pixel_size_um[0], field.x_um[0], field.x_um[-1]),
                 np.clip(origin[1]+(j+.5)*field.pixel_size_um[1], field.y_um[0], field.y_um[-1])]
        full[i:i+kx, j:j+ky] += density[i, j]*lookup([point])[0]
    return full[kx//2:kx//2+density.shape[0], ky//2:ky//2+density.shape[1]]


class PSFFieldTests(unittest.TestCase):
    def field(self, shape=(4, 5), grid=([.2, 1.7, 5.], [-.3, 3.8])):
        rng = np.random.default_rng(743)
        p = rng.uniform(.01, 1., (len(grid[0]), len(grid[1]), *shape))
        p /= p.sum(axis=(-2, -1), keepdims=True)
        return PSFField(p, *grid, (.4, .7), 920., {"provider": "asymmetric random test"})

    def test_direct_scatter_odd_even_irregular_grid_and_origin(self):
        a = np.random.default_rng(6).uniform(size=(9, 8))
        for shape in ((4, 5), (5, 4), (6, 6), (13, 11)):
            field = self.field(shape)
            for origin in ((0., 0.), (-1.3, 2.1)):
                with self.subTest(shape=shape, origin=origin):
                    got = render_image(a, field, pixel_size_um=field.pixel_size_um, origin_um=origin)
                    ref = direct_reference(a, field, origin)
                    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)

    def test_impulses_select_source_not_output_kernel(self):
        field = self.field((4, 5))
        for source in ((0, 0), (4, 4), (8, 7)):
            a = np.zeros((9, 8)); a[source] = 1.
            got = render_image(a, field, pixel_size_um=field.pixel_size_um)
            np.testing.assert_allclose(got, direct_reference(a, field), atol=1e-14)

    def test_constant_field_reduces_to_legacy_convolution(self):
        a = np.random.default_rng(3).uniform(size=(17, 19))
        for shape in ((4, 5), (5, 4)):
            field = self.field(shape)
            kernel = field.intensity[0, 0]
            constant = PSFField(np.broadcast_to(kernel, field.intensity.shape), field.x_um,
                                field.y_um, field.pixel_size_um, field.wavelength_nm)
            old = zstack(a[:, :, None], kernel.copy()[:, :, None], device="cpu")[:, :, 0].numpy()
            got = render_image(a, constant, pixel_size_um=constant.pixel_size_um)
            np.testing.assert_allclose(got, old, rtol=1e-12, atol=1e-14)
            single = SampledPSF(kernel, field.pixel_size_um, 920.)
            np.testing.assert_allclose(render_image(a, single, pixel_size_um=field.pixel_size_um), old,
                                       rtol=1e-12, atol=1e-14)

    def test_singleton_axes_and_reusable_plan(self):
        for grid in (([1.], [2.]), ([1.], [-1., 3.]), ([-1., 4.], [2.])):
            field = self.field(grid=grid)
            plan = PSFRenderer(field, (9, 8), pixel_size_um=field.pixel_size_um)
            for a in (np.ones((9, 8)), np.eye(9, 8), np.zeros((9, 8))):
                np.testing.assert_allclose(plan(a), direct_reference(a, field), atol=1e-14)

    def test_interior_flux_and_boundary_loss(self):
        field = self.field((4, 5))
        a = np.zeros((20, 20)); a[10, 10] = 1.
        self.assertAlmostEqual(render_image(a, field, pixel_size_um=field.pixel_size_um).sum(), 1.)
        a[:] = 0.; a[0, 0] = 1.
        self.assertLess(render_image(a, field, pixel_size_um=field.pixel_size_um).sum(), 1.)

    def test_provider_roundtrip_and_old_imports(self):
        from maomi.integrations import SampledPSF as old
        from maomi.integrations.psf import SampledPSF as older
        self.assertIs(old, SampledPSF); self.assertIs(older, SampledPSF)
        field = PSFField.from_provider(lambda x, y: SampledPSF.normalized(
            np.array([[1., 2.+x], [3.+y, 4.]]), (.4, .7), 920., {"x": x, "y": y}), [0., 2.], [0., 3.])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"field.npz"
            field.save(path); back = PSFField.load(path)
            np.testing.assert_array_equal(field.intensity, back.intensity)
            self.assertEqual(field.provenance, back.provenance)
        a = np.ones((9, 8))
        expected = expected_measurement(a, field, pixel_size_um=field.pixel_size_um,
                                        origin_um=(-1., 2.), photons_per_unit=3., background_e=4.)
        np.testing.assert_allclose(expected, 3*direct_reference(a, field, (-1., 2.))+4., atol=1e-13)

    def test_invalid_contracts_are_rejected(self):
        f = self.field()
        for x in ([1., 1., 2.], [3., 2., 1.], [0., np.nan, 2.]):
            with self.assertRaises(ValueError):
                PSFField(f.intensity, x, f.y_um, f.pixel_size_um, f.wavelength_nm)
        for p in (f.intensity*2, -f.intensity, f.intensity*np.nan):
            with self.assertRaises(ValueError):
                PSFField(p, f.x_um, f.y_um, f.pixel_size_um, f.wavelength_nm)
        with self.assertRaises(ValueError):
            PSFField.from_provider(lambda x, y: SampledPSF.normalized(
                np.ones((3, 3)), (1.+x, 1.), 920.), [0., 1.], [0.])
        with self.assertRaises(ValueError):
            PSFRenderer(f, (9, 8), pixel_size_um=(1., 1.))
        plan = PSFRenderer(f, (9, 8), pixel_size_um=f.pixel_size_um)
        for a in (np.ones((9, 9)), -np.ones((9, 8)), np.full((9, 8), np.nan)):
            with self.assertRaises(ValueError):
                plan(a)

    def test_explicit_resampling_preserves_origin_and_records_support(self):
        a = np.zeros((4, 5)); a[2, 2] = 1.
        p = SampledPSF(a, (.4, .7), 920.)
        identity = resample_psf(p, p.pixel_size_um, a.shape)
        np.testing.assert_allclose(identity.intensity, a, atol=1e-14)
        q = resample_psf(p, (.2, .35), (9, 10))
        self.assertEqual(np.unravel_index(q.intensity.argmax(), q.intensity.shape), (4, 5))
        self.assertAlmostEqual(q.intensity.sum(), 1.)
        self.assertIn("pre_normalization_mass_estimate", q.provenance)

    @unittest.skipUnless(importlib.util.find_spec("rayoptics"), "optional RayOptics is not installed")
    def test_real_rayoptics_field_traces(self):
        from rayoptics.raytr.opticalspec import FieldSpec
        from maomi.integrations.rayoptics import demo_lens, rayoptics_psf, rayoptics_psf_field
        model = demo_lens()
        spec = model["optical_spec"]
        spec["fov"] = FieldSpec(spec, key=["object", "height"], flds=[0., .05, .10],
                               value=.10, is_relative=False)
        model.update_model()
        pitch = rayoptics_psf(model, fft_size=128).pixel_size_um
        field = rayoptics_psf_field(model, [0.], [0., 50., 100.], [[0, 1, 2]],
                                    pixel_size_um=pitch, shape=(128, 128), fft_size=128)
        self.assertEqual(field.intensity.shape, (1, 3, 128, 128))
        self.assertGreater(np.linalg.norm(field.intensity[0, 0]-field.intensity[0, 2]), 1e-8)
        for i, record in enumerate(field.provenance["samples"]):
            self.assertEqual(record["provider"]["source"]["field_index"], i)
        a = np.zeros((5, 70)); a[2, [10, 35, 60]] = 1.
        got = render_image(a, field, pixel_size_um=field.pixel_size_um)
        np.testing.assert_allclose(got, direct_reference(a, field), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
