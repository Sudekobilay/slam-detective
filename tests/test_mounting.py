"""Montaj dönüşümlerinin analitik doğrulaması."""

import math
import unittest

from simulator.environment import create_demo_environment
from simulator.lidar import Lidar, LidarConfig
from simulator.mounting import SensorMount
from simulator.robot import Pose


class TestSensorMount(unittest.TestCase):

    def assert_point_close(self, actual, expected):
        self.assertAlmostEqual(
            actual[0], expected[0], delta=1e-9
        )
        self.assertAlmostEqual(
            actual[1], expected[1], delta=1e-9
        )

    def test_center_mount(self):
        mount = SensorMount()
        robot_pose = Pose(2.0, 3.0, 0.4)

        sensor_pose = mount.world_pose(robot_pose)

        self.assertEqual(sensor_pose, robot_pose)

    def test_forward_offset(self):
        mount = SensorMount(x=0.25)
        sensor_pose = mount.world_pose(
            Pose(1.0, 3.0, 0.0)
        )

        self.assert_point_close(
            (sensor_pose.x, sensor_pose.y),
            (1.25, 3.0),
        )

    def test_offset_rotates_with_robot(self):
        mount = SensorMount(x=0.25)
        sensor_pose = mount.world_pose(
            Pose(1.0, 3.0, math.pi / 2)
        )

        self.assert_point_close(
            (sensor_pose.x, sensor_pose.y),
            (1.0, 3.25),
        )

    def test_mount_yaw_adds_to_robot_heading(self):
        mount = SensorMount(yaw=math.pi / 2)
        sensor_pose = mount.world_pose(
            Pose(theta=math.pi / 4)
        )

        self.assertAlmostEqual(
            sensor_pose.theta,
            3 * math.pi / 4,
            delta=1e-9,
        )

    def test_sensor_point_to_robot(self):
        # Sensör robotta (0.5, 0.2) noktasında,
        # sola bakıyor. Sensörün 2 m önündeki
        # nokta robot koordinatında (0.5, 2.2).
        mount = SensorMount(
            x=0.5,
            y=0.2,
            yaw=math.pi / 2,
        )

        self.assert_point_close(
            mount.point_to_robot((2.0, 0.0)),
            (0.5, 2.2),
        )

    def test_sensor_point_to_world(self):
        mount = SensorMount(
            x=0.5,
            y=0.2,
            yaw=math.pi / 2,
        )
        robot_pose = Pose(
            x=2.0,
            y=3.0,
            theta=math.pi / 2,
        )

        # Robot koordinatındaki (0.5, 2.2),
        # dünyada (-0.2, 3.5) noktasına dönüşür.
        self.assert_point_close(
            mount.point_to_world(
                (2.0, 0.0),
                robot_pose,
            ),
            (-0.2, 3.5),
        )

    def test_forward_mount_changes_measured_distance(self):
        environment = create_demo_environment()
        sensor = Lidar(LidarConfig(beam_count=1))
        robot_pose = Pose(1.0, 3.0, 0.0)

        center_mount = SensorMount()
        forward_mount = SensorMount(x=0.25)

        center_scan = sensor.scan(
            center_mount.world_pose(robot_pose),
            environment,
        )
        forward_scan = sensor.scan(
            forward_mount.world_pose(robot_pose),
            environment,
        )

        self.assertAlmostEqual(
            center_scan.ranges[0], 2.0
        )
        self.assertAlmostEqual(
            forward_scan.ranges[0], 1.75
        )

    def test_invalid_mount_rejected(self):
        with self.assertRaises(ValueError):
            SensorMount(x=math.inf)

        with self.assertRaises(ValueError):
            SensorMount(yaw=math.nan)


if __name__ == "__main__":
    unittest.main()