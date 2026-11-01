"""İdeal LiDAR için bağımsız geometrik doğrulamalar."""

import math
import unittest

from simulator.environment import (
    Wall,
    create_demo_environment,
)
from simulator.lidar import (
    Lidar,
    LidarConfig,
    ray_wall_distance,
)
from simulator.robot import Pose


class TestLidar(unittest.TestCase):

    def setUp(self):
        self.environment = create_demo_environment()

    def test_four_cardinal_directions(self):
        sensor = Lidar(LidarConfig(beam_count=4))
        scan = sensor.scan(
            Pose(1.0, 1.0, 0.0),
            self.environment,
        )

        expected = (1.0, 1.0, 9.0, 7.0)

        for actual, target in zip(scan.ranges, expected):
            self.assertIsNotNone(actual)
            self.assertAlmostEqual(actual, target)

    def test_sensor_rotation(self):
        # Tek ışın sensörün ileri yönüne bakar.
        # 90° dönüş sonrası üst duvar 7 m uzaktadır.
        sensor = Lidar(LidarConfig(beam_count=1))
        scan = sensor.scan(
            Pose(1.0, 1.0, math.pi / 2),
            self.environment,
        )

        self.assertAlmostEqual(scan.ranges[0], 7.0)

    def test_nearest_obstacle_blocks_far_wall(self):
        # Robot (1,3), ışın +x yönünde.
        # Engel x=3'te: 2 m; dış duvar x=10'da: 9 m.
        sensor = Lidar(LidarConfig(beam_count=1))
        scan = sensor.scan(
            Pose(1.0, 3.0, 0.0),
            self.environment,
        )

        self.assertAlmostEqual(scan.ranges[0], 2.0)

    def test_diagonal_obstacle_distance(self):
        # (1,1)'den 45° ışın, engeli (3,3)'te görür.
        # Mesafe sqrt(2² + 2²) = sqrt(8).
        sensor = Lidar(LidarConfig(beam_count=1))
        scan = sensor.scan(
            Pose(1.0, 1.0, math.pi / 4),
            self.environment,
        )

        self.assertAlmostEqual(
            scan.ranges[0], math.sqrt(8.0)
        )

    def test_no_return_beyond_max_range(self):
        sensor = Lidar(
            LidarConfig(beam_count=4, max_range=0.5)
        )
        scan = sensor.scan(
            Pose(1.0, 1.0, 0.0),
            self.environment,
        )

        self.assertEqual(
            scan.ranges, (None, None, None, None)
        )
        self.assertEqual(
            scan.points, (None, None, None, None)
        )

    def test_surface_exactly_at_max_range(self):
        # Sola bakan sensör ile x=0 duvarı arası 1 m.
        sensor = Lidar(
            LidarConfig(beam_count=1, max_range=1.0)
        )
        scan = sensor.scan(
            Pose(1.0, 1.0, math.pi),
            self.environment,
        )

        self.assertIsNotNone(scan.ranges[0])
        self.assertAlmostEqual(scan.ranges[0], 1.0)

    def test_full_circle_has_no_duplicate_direction(self):
        sensor = Lidar(LidarConfig(beam_count=4))
        angles = sensor.beam_angles()

        self.assertEqual(len(angles), 4)

        expected = (
            -math.pi,
            -math.pi / 2,
            0.0,
            math.pi / 2,
        )

        for actual, target in zip(angles, expected):
            self.assertAlmostEqual(actual, target)

    def test_partial_field_of_view(self):
        sensor = Lidar(
            LidarConfig(
                field_of_view=math.pi,
                beam_count=3,
            )
        )

        expected = (-math.pi / 2, 0.0, math.pi / 2)

        for actual, target in zip(
            sensor.beam_angles(), expected
        ):
            self.assertAlmostEqual(actual, target)

    def test_points_remain_in_sensor_coordinates(self):
        # Sensör dünyada yukarı baksa da kendi
        # koordinatlarında ileri yön +x'tir.
        sensor = Lidar(LidarConfig(beam_count=1))
        scan = sensor.scan(
            Pose(1.0, 1.0, math.pi / 2),
            self.environment,
        )

        point = scan.points[0]
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 7.0)
        self.assertAlmostEqual(point[1], 0.0)

    def test_wall_behind_ray_is_not_detected(self):
        wall = Wall((-2.0, -1.0), (-2.0, 1.0))

        distance = ray_wall_distance(
            origin=(0.0, 0.0),
            direction=(1.0, 0.0),
            wall=wall,
        )

        self.assertIsNone(distance)

    def test_parallel_and_collinear_walls(self):
        with self.subTest(case="parallel"):
            distance = ray_wall_distance(
                origin=(0.0, 0.0),
                direction=(1.0, 0.0),
                wall=Wall((2.0, 1.0), (4.0, 1.0)),
            )
            self.assertIsNone(distance)

        with self.subTest(case="collinear"):
            distance = ray_wall_distance(
                origin=(0.0, 0.0),
                direction=(1.0, 0.0),
                wall=Wall((2.0, 0.0), (4.0, 0.0)),
            )
            self.assertAlmostEqual(distance, 2.0)

    def test_invalid_configuration_rejected(self):
        invalid_parameters = (
            {"beam_count": 0},
            {"beam_count": 3.5},
            {"field_of_view": 0.0},
            {"field_of_view": 3 * math.pi},
            {"max_range": -1.0},
            {"max_range": math.inf},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    LidarConfig(**parameters)


if __name__ == "__main__":
    unittest.main()