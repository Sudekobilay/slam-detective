"""GPS ölçüm ve kesinti modelinin doğrulanması."""

import unittest

from simulator.gps import GPSConfig, GPSSensor
from simulator.robot import Pose


class TestGPS(unittest.TestCase):

    def test_ideal_position(self):
        packet = GPSSensor().measure(Pose(2.0, 3.0), 0.0)

        self.assertTrue(packet.available)
        self.assertEqual((packet.x_m, packet.y_m), (2.0, 3.0))

    def test_known_bias(self):
        sensor = GPSSensor(
            GPSConfig(bias_x_m=0.5, bias_y_m=-0.3)
        )
        packet = sensor.measure(Pose(2.0, 3.0), 0.0)

        self.assertAlmostEqual(packet.x_m, 2.5)
        self.assertAlmostEqual(packet.y_m, 2.7)

    def test_seed_reproduces_noise(self):
        config = GPSConfig(noise_std_m=0.05)
        first = GPSSensor(config, seed=101)
        second = GPSSensor(config, seed=101)

        for index in range(5):
            self.assertEqual(
                first.measure(Pose(), float(index)),
                second.measure(Pose(), float(index)),
            )

    def test_outage_boundaries(self):
        sensor = GPSSensor(
            GPSConfig(outage_start_s=1.0, outage_end_s=3.0)
        )

        self.assertTrue(sensor.measure(Pose(), 0.0).available)

        missing = sensor.measure(Pose(), 1.0)
        self.assertFalse(missing.available)
        self.assertIsNone(missing.x_m)
        self.assertIsNone(missing.y_m)

        self.assertTrue(sensor.measure(Pose(), 3.0).available)

    def test_outage_does_not_shift_noise_stream(self):
        normal = GPSSensor(
            GPSConfig(noise_std_m=0.05), seed=101
        )
        outage = GPSSensor(
            GPSConfig(
                noise_std_m=0.05,
                outage_start_s=1.0,
                outage_end_s=3.0,
            ),
            seed=101,
        )

        for time_s in (0.0, 1.0, 2.0, 3.0):
            normal_packet = normal.measure(Pose(), time_s)
            outage_packet = outage.measure(Pose(), time_s)

        self.assertEqual(normal_packet, outage_packet)

    def test_packet_contains_no_truth(self):
        packet = GPSSensor().measure(Pose(), 0.0)

        self.assertEqual(
            set(vars(packet)),
            {
                "time_s", "available", "x_m", "y_m",
                "nominal_variance_m2",
            },
        )

    def test_duplicate_time_rejected(self):
        sensor = GPSSensor()
        sensor.measure(Pose(), 0.0)

        with self.assertRaises(ValueError):
            sensor.measure(Pose(), 0.0)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            GPSConfig(noise_std_m=-1.0)

        with self.assertRaises(ValueError):
            GPSConfig(outage_start_s=1.0)

        with self.assertRaises(ValueError):
            GPSConfig(outage_start_s=3.0, outage_end_s=1.0)


if __name__ == "__main__":
    unittest.main()