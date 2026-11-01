"""Robot hareket modelinin analitik doğrulama testleri."""

import math
import unittest

from simulator.robot import (
    DifferentialDriveRobot,
    Pose,
    RobotGeometry,
)


class TestRobotMotion(unittest.TestCase):

    def assert_pose_close(
        self,
        actual: Pose,
        expected_x: float,
        expected_y: float,
        expected_theta: float,
    ):
        """Konumu ve yönü sayısal toleransla karşılaştırır."""
        self.assertAlmostEqual(
            actual.x, expected_x, delta=1e-9
        )
        self.assertAlmostEqual(
            actual.y, expected_y, delta=1e-9
        )

        # π ile -π gibi eşdeğer yönleri doğru karşılaştırır.
        angle_difference = actual.theta - expected_theta
        angle_error = math.atan2(
            math.sin(angle_difference),
            math.cos(angle_difference),
        )

        self.assertAlmostEqual(
            angle_error, 0.0, delta=1e-9
        )

    def test_straight_motion(self):
        robot = DifferentialDriveRobot()
        pose = robot.step(4.0, 4.0, 5.0)

        self.assert_pose_close(pose, 1.0, 0.0, 0.0)

    def test_reverse_motion(self):
        robot = DifferentialDriveRobot()
        pose = robot.step(-4.0, -4.0, 5.0)

        self.assert_pose_close(pose, -1.0, 0.0, 0.0)

    def test_in_place_turn(self):
        robot = DifferentialDriveRobot()
        pose = robot.step(-3.0, 3.0, math.pi / 2)

        self.assert_pose_close(
            pose, 0.0, 0.0, math.pi / 2
        )

    def test_quarter_circle(self):
        robot = DifferentialDriveRobot()
        pose = robot.step(17.0, 23.0, math.pi / 2)

        self.assert_pose_close(
            pose, 1.0, 1.0, math.pi / 2
        )

    def test_full_circle(self):
        robot = DifferentialDriveRobot()
        pose = robot.step(17.0, 23.0, 2 * math.pi)

        self.assert_pose_close(pose, 0.0, 0.0, 0.0)

    def test_square_path(self):
        robot = DifferentialDriveRobot()

        for _ in range(4):
            robot.step(4.0, 4.0, 5.0)
            robot.step(-3.0, 3.0, math.pi / 2)

        self.assert_pose_close(
            robot.pose, 0.0, 0.0, 0.0
        )

    def test_rotated_start(self):
        robot = DifferentialDriveRobot(
            initial_pose=Pose(
                x=2.0, y=3.0, theta=math.pi / 2
            )
        )
        pose = robot.step(4.0, 4.0, 5.0)

        self.assert_pose_close(
            pose, 2.0, 4.0, math.pi / 2
        )

    def test_small_steps_match_single_step(self):
        single_step_robot = DifferentialDriveRobot()
        multiple_steps_robot = DifferentialDriveRobot()

        expected = single_step_robot.step(
            17.0, 23.0, 2.0
        )

        for _ in range(200):
            multiple_steps_robot.step(
                17.0, 23.0, 0.01
            )

        self.assert_pose_close(
            multiple_steps_robot.pose,
            expected.x,
            expected.y,
            expected.theta,
        )

    def test_zero_duration(self):
        initial_pose = Pose(1.0, 2.0, 0.3)
        robot = DifferentialDriveRobot(
            initial_pose=initial_pose
        )

        pose = robot.step(4.0, 4.0, 0.0)

        self.assertEqual(pose, initial_pose)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            RobotGeometry(wheelbase=0.0)

        with self.assertRaises(ValueError):
            Pose(theta=math.nan)

        robot = DifferentialDriveRobot()

        with self.assertRaises(ValueError):
            robot.step(4.0, 4.0, -1.0)

        with self.assertRaises(ValueError):
            robot.step(math.inf, 4.0, 1.0)


if __name__ == "__main__":
    unittest.main()