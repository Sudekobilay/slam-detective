"""Doğru çıkarımının bilinen geometrilerle kontrolü."""

import math
import unittest
import numpy as np

from perception.line_features import LineConfig, extract_lines


class TestLineFeatures(unittest.TestCase):

    def test_vertical_line(self):
        points = [
            (2.0, float(y))
            for y in np.linspace(-1.0, 1.0, 12)
        ]

        lines = extract_lines(points)

        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0].rho_m, 2.0)
        self.assertAlmostEqual(lines[0].length_m, 2.0)
        self.assertAlmostEqual(lines[0].normal[0], 1.0)
        self.assertAlmostEqual(lines[0].normal[1], 0.0)

    def test_oblique_line(self):
        points = [
            (float(x), float(0.5 * x + 1.0))
            for x in np.linspace(0.0, 2.0, 12)
        ]

        lines = extract_lines(points)

        self.assertEqual(len(lines), 1)

        for point in points:
            distance = abs(
                lines[0].normal[0] * point[0]
                + lines[0].normal[1] * point[1]
                - lines[0].rho_m
            )
            self.assertLess(distance, 1e-9)

    def test_corner_splits_into_two_lines(self):
        vertical = [
            (2.0, float(y))
            for y in np.linspace(-1.0, 0.0, 11)
        ]
        horizontal = [
            (float(x), 0.0)
            for x in np.linspace(2.1, 3.0, 10)
        ]

        lines = extract_lines(vertical + horizontal)

        self.assertEqual(len(lines), 2)

    def test_missing_return_separates_groups(self):
        first = [
            (2.0, float(y))
            for y in np.linspace(0.0, 1.0, 11)
        ]
        second = [
            (2.0, float(y))
            for y in np.linspace(1.1, 2.1, 11)
        ]

        lines = extract_lines(first + [None] + second)

        self.assertEqual(len(lines), 2)
        self.assertLess(lines[0].last_beam, lines[1].first_beam)

    def test_identical_points_are_not_a_segment(self):
        self.assertEqual(
            extract_lines([(2.0, 1.0)] * 10),
            [],
        )

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            LineConfig(max_gap_m=-1.0)

        with self.assertRaises(ValueError):
            LineConfig(min_points=2)

    def test_nonfinite_point_rejected(self):
        with self.assertRaises(ValueError):
            extract_lines([(math.nan, 1.0)])


if __name__ == "__main__":
    unittest.main()