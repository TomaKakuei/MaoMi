import importlib.util
import unittest
import numpy as np
from maomi.integrations.volume import PSFVolume,volume_psf_field
from maomi.rana_addon import RanaOperator


class VolumeProviderTests(unittest.TestCase):
    def test_complex_polarization_power_and_relative_gain(self):
        field=np.zeros((3,3,3,2),complex)
        field[1,1,1]=[1,1j]
        def provider(x,y):
            return PSFVolume(field*(1+x),(1,1,1),920,"complex_field")
        bank=volume_psf_field(provider,[0,1],[0,1],photon_order=2,normalization="reference_sum")
        self.assertEqual(bank.kernels[0,0].sum(),1.)
        self.assertEqual(bank.kernels[1,0].sum(),16.)
        effective=PSFVolume(np.ones((2,2,2)),(1,1,1),920,"effective_excitation")
        np.testing.assert_array_equal(effective.effective(3),np.ones((2,2,2)))

    @unittest.skipUnless(importlib.util.find_spec("rayoptics"),"Install the optics extra")
    def test_rayoptics_xyz_and_native_pupil(self):
        from maomi.integrations.rayoptics import demo_lens
        from maomi.integrations.rayoptics_volume import rayoptics_volume_field,rayoptics_native_pupil
        from maomi import OpticalConfig
        model=demo_lens(wavelength_nm=920)
        field=rayoptics_volume_field(model,[0,8],[0,8],np.zeros((2,2),int),
            voxel_size_um=(1,1,1),shape=(4,4,3),num_rays=9,fft_size=32)
        self.assertEqual(field.kernels.shape,(2,2,4,4,3))
        self.assertTrue(np.isfinite(RanaOperator(field).scan_slab(np.ones((8,8,3)))).all())
        cfg=OpticalConfig(volume_um=(8,8,6),depth_um=5,tail_um=2,psf_um=(4,4,4))
        pupil=rayoptics_native_pupil(model,cfg,num_rays=9)
        self.assertTrue(np.isfinite(pupil.numpy()).all())
        self.assertGreater(float(pupil.abs().sum()),0.)


if __name__=="__main__":
    unittest.main()
