"""Robot gövdesi ve hareket boyunca çarpışma kontrolleri."""

import math
import unittest

from simulator.collision import body_clearance, try_step
from simulator.environment import (
    Environment,
    Rectangle,
    create_demo_environment,
)
from simulator.robot import DifferentialDriveRobot, Pose


class TestCollision(unittest.TestCase):

    def setUp(self):
        self.environment = create_demo_environment()

    def test_safe_motion_is_applied(self):
        robot = DifferentialDriveRobot(
            initial_pose=Pose(1.0, 3.0, 0.0)
        )

        accepted = try_step(
            robot, self.environment, 4.0, 4.0, 1.0
        )

        self.assertTrue(accepted)
        self.assertAlmostEqual(robot.pose.x, 1.2)
        self.assertAlmostEqual(robot.pose.y, 3.0)

    def test_blocked_motion_keeps_original_pose(self):
        initial_pose = Pose(1.0, 3.0, 0.0)
        robot = DifferentialDriveRobot(
            initial_pose=initial_pose
        )

        accepted = try_step(
            robot, self.environment, 4.0, 4.0, 10.0
        )

        self.assertFalse(accepted)
        self.assertEqual(robot.pose, initial_pose)

    def test_thin_obstacle_cannot_be_skipped(self):
        # Başlangıç ve bitiş güvenli; arada ince engel var.
        environment = Environment(
            boundary=Rectangle(0.0, 0.0, 10.0, 8.0),
            obstacles=(
                Rectangle(2.0, 2.0, 2.01, 4.0),
            ),
        )
        initial_pose = Pose(1.0, 3.0, 0.0)
        robot = DifferentialDriveRobot(
            initial_pose=initial_pose
        )

        # Engelsiz hareket hesabıyla bitiş x=4 olurdu.
        self.assertGreater(
            body_clearance(
                Pose(4.0, 3.0, 0.0), environment
            ),
            0.0,
        )

        accepted = try_step(
            robot, environment, 4.0, 4.0, 15.0
        )

        self.assertFalse(accepted)
        self.assertEqual(robot.pose, initial_pose)

    def test_safe_in_place_turn(self):
        robot = DifferentialDriveRobot(
            initial_pose=Pose(1.0, 3.0, 0.0)
        )

        accepted = try_step(
            robot,
            self.environment,
            -3.0,
            3.0,
            math.pi / 2,
        )

        self.assertTrue(accepted)
        self.assertAlmostEqual(robot.pose.x, 1.0)
        self.assertAlmostEqual(robot.pose.y, 3.0)
        self.assertAlmostEqual(
            robot.pose.theta, math.pi / 2
        )

    def test_contact_counts_as_collision(self):
        clearance = body_clearance(
            Pose(1.0, 0.35, 0.0),
            self.environment,
        )

        self.assertAlmostEqual(clearance, 0.0)

    def test_start_inside_obstacle_rejected(self):
        robot = DifferentialDriveRobot(
            initial_pose=Pose(3.5, 3.0, 0.0)
        )

        with self.assertRaises(ValueError):
            try_step(
                robot, self.environment, 4.0, 4.0, 1.0
            )

    def test_corner_uses_euclidean_distance(self):
        # En yakın engel köşesi (3,2).
        # Merkez-köşe uzaklığı sqrt(0.3² + 0.3²).
        clearance = body_clearance(
            Pose(2.7, 1.7, 0.0),
            self.environment,
        )

        expected = math.hypot(0.3, 0.3) - 0.35

        self.assertAlmostEqual(clearance, expected)
        self.assertGreater(clearance, 0.0)

    def test_invalid_parameters_rejected(self):
        robot = DifferentialDriveRobot(
            initial_pose=Pose(1.0, 3.0, 0.0)
        )

        with self.assertRaises(ValueError):
            body_clearance(
                robot.pose, self.environment, radius=0.0
            )

        with self.assertRaises(ValueError):
            try_step(
                robot, self.environment, 4.0, 4.0, -1.0
            )

        with self.assertRaises(ValueError):
            try_step(
                robot,
                self.environment,
                4.0,
                4.0,
                1.0,
                max_center_travel=0.0,
            )


if __name__ == "__main__":
    unittest.main()