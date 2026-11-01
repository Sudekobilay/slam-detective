"""Gyro modeli ve yön integrasyonunun doğrulanması."""

import unittest

from estimation.gyro_heading import GyroHeading
from simulator.imu import GyroConfig, Gyroscope


class TestGyro(unittest.TestCase):

    def test_ideal_stationary(self):
        gyro = Gyroscope()
        heading = GyroHeading()

        for _ in range(10):
            heading.update(gyro.measure(0.0, 0.1))

        self.assertAlmostEqual(heading.theta, 0.0)

    def test_signed_rotation(self):
        for rate in (-0.3, 0.3):
            with self.subTest(rate=rate):
                gyro = Gyroscope()
                heading = GyroHeading()

                for _ in range(10):
                    heading.update(gyro.measure(rate, 0.1))

                self.assertAlmostEqual(heading.theta, rate)

    def test_bias_accumulates_with_time(self):
        gyro = Gyroscope(GyroConfig(bias_rad_s=0.03))
        heading = GyroHeading()

        for _ in range(40):
            heading.update(gyro.measure(0.0, 0.1))

        # 0.03 rad/s × 4 s = 0.12 rad.
        self.assertAlmostEqual(heading.theta, 0.12)

    def test_seed_reproduces_noise(self):
        config = GyroConfig(noise_std_rad_s=0.01)
        first = Gyroscope(config, seed=101)
        second = Gyroscope(config, seed=101)

        for _ in range(20):
            self.assertEqual(
                first.measure(0.3, 0.1),
                second.measure(0.3, 0.1),
            )

    def test_packet_contains_no_truth_or_bias(self):
        packet = Gyroscope().measure(0.3, 0.1)

        self.assertEqual(
            set(vars(packet)),
            {
                "time_s",
                "dt_s",
                "yaw_rate_rad_s",
                "nominal_variance",
            },
        )

    def test_duplicate_packet_rejected(self):
        heading = GyroHeading()
        packet = Gyroscope().measure(0.3, 0.1)

        heading.update(packet)
        previous_heading = heading.theta

        with self.assertRaises(ValueError):
            heading.update(packet)

        self.assertEqual(heading.theta, previous_heading)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            GyroConfig(noise_std_rad_s=-0.1)

        with self.assertRaises(ValueError):
            Gyroscope().measure(0.0, 0.0)


if __name__ == "__main__":
    unittest.main()