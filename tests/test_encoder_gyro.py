"""Birleşik hareket tahmininin analitik kontrolleri."""

import math
import unittest

from estimation.encoder_gyro import EncoderGyroOdometry
from estimation.odometry import WheelOdometry
from simulator.encoder import EncoderConfig, EncoderMeasurement
from simulator.imu import GyroMeasurement


class TestEncoderGyro(unittest.TestCase):

    def packet_pair(self):
        encoder = EncoderMeasurement(
            time_s=1.0,
            dt_s=1.0,
            left_delta_counts=25,
            right_delta_counts=25,
        )
        gyro = GyroMeasurement(
            time_s=1.0,
            dt_s=1.0,
            yaw_rate_rad_s=1.0,
            nominal_variance=0.0,
        )
        return encoder, gyro

    def test_zero_weight_matches_encoder(self):
        config = EncoderConfig(counts_per_revolution=100)
        combined = EncoderGyroOdometry(
            gyro_weight=0.0,
            encoder_config=config,
        )
        baseline = WheelOdometry(encoder_config=config)
        encoder, gyro = self.packet_pair()

        actual = combined.update(encoder, gyro)
        expected = baseline.update(encoder)

        self.assertAlmostEqual(actual.x, expected.x)
        self.assertAlmostEqual(actual.y, expected.y)
        self.assertAlmostEqual(actual.theta, expected.theta)

    def test_full_weight_uses_gyro_turn(self):
        estimator = EncoderGyroOdometry(gyro_weight=1.0)
        encoder, gyro = self.packet_pair()

        pose = estimator.update(encoder, gyro)

        self.assertAlmostEqual(pose.theta, 1.0)

    def test_half_weight_analytical_arc(self):
        config = EncoderConfig(counts_per_revolution=100)
        estimator = EncoderGyroOdometry(
            gyro_weight=0.5,
            encoder_config=config,
        )
        encoder, gyro = self.packet_pair()

        pose = estimator.update(encoder, gyro)

        # İlerleme pi/40 m, toplam dönüş 0.5 rad.
        ds = math.pi / 40.0
        radius = ds / 0.5

        self.assertAlmostEqual(
            pose.x, radius * math.sin(0.5)
        )
        self.assertAlmostEqual(
            pose.y, radius * (1.0 - math.cos(0.5))
        )
        self.assertAlmostEqual(pose.theta, 0.5)

    def test_mismatched_intervals_rejected(self):
        estimator = EncoderGyroOdometry()
        encoder, _ = self.packet_pair()
        gyro = GyroMeasurement(0.5, 0.5, 1.0, 0.0)
        previous_pose = estimator.pose

        with self.assertRaises(ValueError):
            estimator.update(encoder, gyro)

        self.assertEqual(estimator.pose, previous_pose)

    def test_duplicate_pair_rejected(self):
        estimator = EncoderGyroOdometry()
        encoder, gyro = self.packet_pair()
        estimator.update(encoder, gyro)
        previous_pose = estimator.pose

        with self.assertRaises(ValueError):
            estimator.update(encoder, gyro)

        self.assertEqual(estimator.pose, previous_pose)

    def test_invalid_weight_rejected(self):
        for weight in (-0.1, 1.1, math.nan):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    EncoderGyroOdometry(gyro_weight=weight)


if __name__ == "__main__":
    unittest.main()