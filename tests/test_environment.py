"""Ortam geometrisinin doğrulama testleri."""

import math
import unittest

from simulator.environment import (
    Environment,
    Rectangle,
    Wall,
    create_demo_environment,
)


class TestEnvironment(unittest.TestCase):

    def test_wall_length(self):
        # 3-4-5 üçgeninden bağımsız uzunluk kontrolü.
        wall = Wall((0.0, 0.0), (3.0, 4.0))

        self.assertAlmostEqual(wall.length, 5.0)

    def test_zero_length_wall_rejected(self):
        with self.assertRaises(ValueError):
            Wall((1.0, 2.0), (1.0, 2.0))

    def test_nonfinite_wall_rejected(self):
        with self.assertRaises(ValueError):
            Wall((0.0, 0.0), (math.inf, 1.0))

    def test_rectangle_forms_closed_boundary(self):
        walls = Rectangle(1.0, 2.0, 4.0, 6.0).walls()

        self.assertEqual(len(walls), 4)

        for index, wall in enumerate(walls):
            next_wall = walls[(index + 1) % len(walls)]
            self.assertEqual(wall.end, next_wall.start)

        self.assertAlmostEqual(
            sum(wall.length for wall in walls),
            14.0,
        )

    def test_invalid_rectangle_rejected(self):
        invalid_values = (
            (0.0, 0.0, 0.0, 1.0),
            (2.0, 0.0, 1.0, 1.0),
            (0.0, 2.0, 1.0, 1.0),
            (0.0, 0.0, math.nan, 1.0),
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    Rectangle(*values)

    def test_empty_room_has_four_walls(self):
        environment = Environment(
            boundary=Rectangle(0.0, 0.0, 10.0, 8.0)
        )

        self.assertEqual(len(environment.walls), 4)

    def test_obstacle_outside_room_rejected(self):
        with self.assertRaises(ValueError):
            Environment(
                boundary=Rectangle(0.0, 0.0, 10.0, 8.0),
                obstacles=(
                    Rectangle(9.0, 2.0, 11.0, 3.0),
                ),
            )

    def test_obstacle_touching_boundary_rejected(self):
        with self.assertRaises(ValueError):
            Environment(
                boundary=Rectangle(0.0, 0.0, 10.0, 8.0),
                obstacles=(
                    Rectangle(0.0, 2.0, 1.0, 3.0),
                ),
            )

    def test_demo_environment_geometry(self):
        environment = create_demo_environment()

        self.assertEqual(len(environment.obstacles), 2)
        self.assertEqual(len(environment.walls), 12)

        self.assertAlmostEqual(
            sum(
                wall.length
                for wall in environment.boundary.walls()
            ),
            36.0,
        )

        self.assertAlmostEqual(
            sum(wall.length for wall in environment.walls),
            49.0,
        )


if __name__ == "__main__":
    unittest.main()