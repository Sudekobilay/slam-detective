"""ICP'nin bilinen geometrik dönüşümlerle doğrulanması."""

import math
import unittest
import numpy as np

from estimation.scan_matching import icp_2d


class TestScanMatching(unittest.TestCase):

    def setUp(self):
        self.points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.7, 0.3],
            [0.4, 2.0],
            [2.0, 2.0],
            [2.4, 1.1],
            [1.0, 3.0],
        ])

    def test_identical_clouds(self):
        result = icp_2d(
            self.points, self.points, trim_fraction=1.0
        )

        self.assertTrue(result["converged"])
        self.assertLess(result["rmse_m"], 1e-9)
        np.testing.assert_allclose(
            result["rotation"], np.eye(2), atol=1e-9
        )

    def test_known_rotation_and_translation(self):
        angle = 0.07
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        translation = np.array([0.08, -0.04])

        # target = source @ R.T + t
        source = (self.points - translation) @ rotation

        result = icp_2d(
            source, self.points, trim_fraction=1.0
        )

        self.assertTrue(result["converged"])
        np.testing.assert_allclose(
            result["rotation"], rotation, atol=1e-8
        )
        np.testing.assert_allclose(
            result["translation_m"], translation, atol=1e-8
        )

    def test_disjoint_clouds_report_failure(self):
        result = icp_2d(
            self.points + 100.0,
            self.points,
        )

        self.assertFalse(result["converged"])
        self.assertEqual(result["reason"], "not_enough_matches")
        self.assertIsNone(result["rmse_m"])

    def test_nonfinite_points_rejected(self):
        points = self.points.copy()
        points[0, 0] = math.nan

        with self.assertRaises(ValueError):
            icp_2d(points, self.points)

    def test_inputs_unchanged(self):
        original = self.points.copy()

        icp_2d(self.points, self.points)

        np.testing.assert_array_equal(self.points, original)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            icp_2d(
                self.points, self.points,
                initial_translation=(math.nan, 0.0),
            )

        with self.assertRaises(ValueError):
            icp_2d(
                self.points, self.points,
                max_distance=-1.0,
            )


if __name__ == "__main__":
    unittest.main()