import unittest
import numpy as np

from maomi.movie import row_motion, sample_motion


class MotionBoundaryTests(unittest.TestCase):
    def test_random_motion_stays_inside_retained_field(self):
        # This seed previously reached an invalid final column at frame 17.
        shape = (264, 264, 16)
        image = np.arange(shape[0]*shape[1], dtype=np.float32).reshape(shape[:2])
        motion = sample_motion(shape, 6, 256, buffer=4,
                               rng=np.random.default_rng(44))
        for frame in motion:
            shifted = row_motion(image, 4, frame["x"], frame["y"])
            self.assertEqual(shifted.shape, (256, 256))
            self.assertTrue(np.isfinite(shifted).all())
            self.assertGreaterEqual(frame["z"], 1)
            self.assertLessEqual(frame["z"]+6-1, shape[2])

    def test_zero_buffer_does_not_create_missing_border(self):
        image = np.arange(32*28, dtype=np.float32).reshape(32, 28)
        for enabled in (False, True):
            motion = sample_motion((32,28,12), 4, 16, buffer=0, enabled=enabled,
                                   rng=np.random.default_rng(17))
            for frame in motion:
                np.testing.assert_array_equal(
                    row_motion(image, 0, frame["x"], frame["y"]), image)


if __name__ == "__main__":
    unittest.main()
