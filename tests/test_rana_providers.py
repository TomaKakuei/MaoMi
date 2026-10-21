"""Third-party providers compose with the core, native host and archives."""
from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest
import numpy as np

from maomi.rana import (FullFieldPSF, DepthPSFBank, PSFProvider, RanaAddon,
                        RanaOperator, attach_rana, resolve_field)
from maomi.archive import save_optics, load_optics


def local_field(gain=1., operator='excitation'):
    kernels = np.zeros((2, 2, 3, 3, 3))
    kernels[..., 1, 1, 1] = gain*np.array([[1., 2.], [3., 4.]])
    return FullFieldPSF(kernels, [0, 6], [0, 8], (1, 1, 1), 920, operator,
                        {'solver': 'external-test-provider', 'gain': gain})


class ExternalProvider:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def build_field(self):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError('Provider rerun after preparation')
        return self.result


class ProviderContract(unittest.TestCase):
    def test_core_prepares_once_across_frames_and_cache_reset(self):
        field = local_field()
        provider = ExternalProvider(field)
        self.assertIsInstance(provider, PSFProvider)
        op = RanaOperator(provider)
        volume = np.arange(6*8*3).reshape(6, 8, 3).astype(float)
        # Center-only kernels are a position-dependent gain, independently
        # computable without an FFT or the provider implementation.
        x, y = (np.arange(6)+.5)/6, (np.arange(8)+.5)/8
        x, y = x*x*(3-2*x), y*y*(3-2*y)
        gain = 1+2*x[:, None]+y[None, :]
        expected = volume*gain[..., None]
        for _ in range(2):
            np.testing.assert_allclose(op.static(volume), expected, atol=1e-12)
            np.testing.assert_allclose(op.scan_slab(volume), expected[..., 1], atol=1e-12)
            op.clear_cache()
        self.assertEqual(provider.calls, 1)

    def test_host_and_archive_store_resolved_fields_without_provider(self):
        bank = DepthPSFBank([0, 4], (local_field(1, 'detection'), local_field(2, 'detection')))
        provider = ExternalProvider(bank)
        mask = np.full((6, 8), .7)
        native = {'mask': mask, 'settings': {'source': 'test'}}
        wrapped = attach_rana(native, RanaAddon(field=provider, voxel_size_um=(1, 1, 1)))
        self.assertIs(wrapped['mask'], mask)
        volume = np.ones((6, 8, 3))
        expected = wrapped['rana'].static(volume)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'optics.h5'
            save_optics(wrapped, path)
            restored = load_optics(path)['rana']
            np.testing.assert_array_equal(restored.static(volume), expected)
            self.assertEqual(restored.field.provenance, bank.fields[0].provenance)
        self.assertEqual(provider.calls, 1)

    def test_readonly_channel_mapping_selects_independent_providers(self):
        fields = MappingProxyType({'a': ExternalProvider(local_field()),
                                   'b': ExternalProvider(local_field(2))})
        native = {'channels': {'a': {}, 'b': {}}, 'beam_weights': {'a': .3, 'b': .7}}
        volume = np.ones((6, 8, 3))
        result = attach_rana(native, RanaAddon(field=fields), volume=volume)
        a, b = [result['channels'][name]['image'] for name in ('a', 'b')]
        np.testing.assert_array_equal(b, 2*a)
        np.testing.assert_array_equal(result['image'], .3*a+.7*b)
        self.assertEqual([p.calls for p in fields.values()], [1, 1])
        with self.assertRaisesRegex(ValueError, 'per optical channel'):
            attach_rana(native, RanaAddon(field={'a': local_field()}))

    def test_invalid_providers_fail_at_preparation(self):
        with self.assertRaisesRegex(TypeError, 'build_field'):
            resolve_field(object())
        with self.assertRaisesRegex(TypeError, 'must return'):
            resolve_field(ExternalProvider(np.ones((3, 3, 3))))
        short = FullFieldPSF(np.ones((2, 2, 3, 3)), [0, 6], [0, 8], (1, 1), 920, 'excitation')
        with self.assertRaisesRegex(TypeError, 'XYZ-kernel'):
            RanaOperator(ExternalProvider(short))
        with self.assertRaisesRegex(ValueError, 'voxel size'):
            attach_rana({}, RanaAddon(field=ExternalProvider(local_field()), voxel_size_um=(2, 2, 2)))

    def test_provider_errors_are_not_hidden_by_fallback(self):
        class FailedSolver:
            def build_field(self):
                raise RuntimeError('external optical solver failed')
        with self.assertRaisesRegex(RuntimeError, 'external optical solver failed'):
            RanaOperator(FailedSolver())


if __name__ == '__main__':
    unittest.main()
