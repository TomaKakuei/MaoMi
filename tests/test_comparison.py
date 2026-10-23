import unittest
import numpy as np
from maomi.comparison import compare_images


class ImageComparisonTests(unittest.TestCase):
    def test_roi_cannot_hide_full_field_difference(self):
        ref = np.ones((20, 20))
        other = ref.copy()
        other[:5] *= 2
        result = compare_images(ref, other, roi=((10, 20), (0, 20)))
        self.assertEqual(result["full_field"]["relative_l2"], .5)
        self.assertEqual(result["roi"]["relative_l2"], 0.)
        self.assertEqual(result["display_range"], [1., 2.])
        self.assertTrue(result["roi"]["identical"])

    def test_zero_reference_and_invalid_regions(self):
        a = np.zeros((2, 3))
        self.assertIsNone(compare_images(a, a+1)["full_field"]["relative_l2"])
        self.assertEqual(compare_images(a, a)["full_field"]["relative_l2"], 0.)
        for roi in (((0, 0), (0, 2)), ((-1, 2), (0, 2)), ((0, 2), (0, 4))):
            with self.assertRaises(ValueError):
                compare_images(a, a, roi=roi)
