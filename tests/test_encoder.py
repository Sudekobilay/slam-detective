"""Encoder kuantalaması ve odometri doğrulaması."""

import math
import unittest

from estimation.odometry import WheelOdometry
from simulator.encoder import (
    EncoderConfig,
    EncoderMeasurement,
    WheelEncoder,
)
from simulator.robot import Pose


class TestEncoderOdometry(unittest.TestCase):

    def test_one_revolution(self):
        encoder = WheelEncoder()

        packet = encoder.measure(
            2.0 * math.pi,
            2.0 * math.pi,
            1.0,
        )

        self.assertEqual(packet.left_delta_counts, 2048)
        self.assertEqual(packet.right_delta_counts, 2048)

    def test_reverse_counts(self):
        encoder = WheelEncoder()

        packet = encoder.measure(
            -2.0 * math.pi,
            -2.0 * math.pi,
            1.0,
        )

        self.assertEqual(packet.left_delta_counts, -2048)
        self.assertEqual(packet.right_delta_counts, -2048)

    def test_fractional_counts_accumulate(self):
        encoder = WheelEncoder(
            EncoderConfig(counts_per_revolution=100)
        )

        # Her ölçümde yalnız 0.1 sayım karşılığı dönüş.
        rate = 0.1 * 2.0 * math.pi / 100
        total = 0

        for _ in range(20):
            packet = encoder.measure(rate, rate, 1.0)
            total += packet.left_delta_counts

        self.assertEqual(total, 2)

    def test_straight_odometry(self):
        config = EncoderConfig(counts_per_revolution=100)
        odometry = WheelOdometry(encoder_config=config)

        pose = odometry.update(
            EncoderMeasurement(
                time_s=1.0,
                dt_s=1.0,
                left_delta_counts=100,
                right_delta_counts=100,
            )
        )

        # 5 cm yarıçaplı tekerleğin bir tam turu.
        self.assertAlmostEqual(
            pose.x, 2.0 * math.pi * 0.05
        )
        self.assertAlmostEqual(pose.y, 0.0)
        self.assertAlmostEqual(pose.theta, 0.0)

    def test_turning_odometry(self):
        config = EncoderConfig(counts_per_revolution=100)
        odometry = WheelOdometry(encoder_config=config)

        pose = odometry.update(
            EncoderMeasurement(
                time_s=1.0,
                dt_s=1.0,
                left_delta_counts=-25,
                right_delta_counts=25,
            )
        )

        self.assertAlmostEqual(pose.x, 0.0)
        self.assertAlmostEqual(pose.y, 0.0)
        self.assertAlmostEqual(pose.theta, math.pi / 6)

    def test_measurement_contains_no_ground_truth(self):
        packet = WheelEncoder().measure(4.0, 4.0, 0.1)

        self.assertEqual(
            set(vars(packet)),
            {
                "time_s",
                "dt_s",
                "left_delta_counts",
                "right_delta_counts",
            },
        )

    def test_duplicate_packet_rejected(self):
        odometry = WheelOdometry(
            initial_pose=Pose(1.0, 3.0, 0.0)
        )
        packet = WheelEncoder().measure(4.0, 4.0, 0.1)

        odometry.update(packet)
        previous_pose = odometry.pose

        with self.assertRaises(ValueError):
            odometry.update(packet)

        self.assertEqual(odometry.pose, previous_pose)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            EncoderConfig(counts_per_revolution=0)

        encoder = WheelEncoder()

        with self.assertRaises(ValueError):
            encoder.measure(4.0, 4.0, 0.0)

        with self.assertRaises(ValueError):
            encoder.measure(math.nan, 4.0, 0.1)

        with self.assertRaises(ValueError):
            EncoderMeasurement(
                time_s=1.0,
                dt_s=1.0,
                left_delta_counts=1.5,
                right_delta_counts=2,
            )


if __name__ == "__main__":
    unittest.main()