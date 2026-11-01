"""Doğru dönüşümü ve eşleştirme kontrolleri."""

from dataclasses import replace
import math
import unittest

from perception.line_association import (
    AssociationConfig,
    associate_lines,
    transform_line,
)
from perception.line_features import LineFeature
from simulator.robot import Pose


def make_line(x=2.0, y_start=-1.0, y_end=1.0):
    return LineFeature(
        normal=(1.0, 0.0),
        rho_m=x,
        start=(x, y_start),
        end=(x, y_end),
        first_beam=0,
        last_beam=4,
        support_count=5,
        length_m=y_end - y_start,
        max_point_error_m=0.0,
    )


class TestLineAssociation(unittest.TestCase):

    def test_line_transformation(self):
        line = transform_line(
            make_line(),
            Pose(1.0, 3.0, math.pi / 2),
        )

        self.assertAlmostEqual(line.normal[0], 0.0)
        self.assertAlmostEqual(line.normal[1], 1.0)
        self.assertAlmostEqual(line.rho_m, 5.0)

        for point, expected in (
            (line.start, (2.0, 5.0)),
            (line.end, (0.0, 5.0)),
        ):
            self.assertAlmostEqual(point[0], expected[0])
            self.assertAlmostEqual(point[1], expected[1])

    def test_opposite_normal_is_same_line(self):
        reference = make_line()
        observation = replace(
            reference,
            normal=(-1.0, 0.0),
            rho_m=-2.0,
        )

        result = associate_lines([observation], [reference])

        self.assertEqual(result[0]["status"], "matched")

    def test_distant_collinear_segments_not_matched(self):
        result = associate_lines(
            [make_line(y_start=4.0, y_end=5.0)],
            [make_line()],
        )

        self.assertEqual(result[0]["status"], "no_candidate")

    def test_similar_candidates_are_ambiguous(self):
        result = associate_lines(
            [make_line()],
            [make_line(x=1.99), make_line(x=2.01)],
        )

        self.assertEqual(result[0]["status"], "ambiguous")
        self.assertIsNone(result[0]["reference_index"])

    def test_reference_used_at_most_once(self):
        result = associate_lines(
            [make_line(), make_line()],
            [make_line()],
        )

        self.assertEqual(
            sum(row["status"] == "matched" for row in result),
            1,
        )
        self.assertEqual(
            sum(row["status"] == "conflict" for row in result),
            1,
        )

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            AssociationConfig(max_gap_m=-1.0)


if __name__ == "__main__":
    unittest.main()