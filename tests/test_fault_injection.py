"""Kontrollü GPS sapmasının izolasyon kontrolleri."""

from copy import deepcopy
import math
import unittest

from simulator.fault_injection import inject_gps_bias


def gps_event(time_s, available=True):
    return {
        "schema_version": 1,
        "sensor_type": "gps",
        "sensor_id": "gps",
        "time_s": time_s,
        "data": {
            "available": available,
            "x_m": 2.0 if available else None,
            "y_m": 3.0 if available else None,
            "nominal_variance_m2": 0.0025,
        },
    }


class TestGPSFaultInjection(unittest.TestCase):

    def test_bias_added_only_to_gps(self):
        encoder = {
            "sensor_type": "encoder",
            "sensor_id": "encoder",
            "time_s": 1.0,
            "data": {"left_delta_counts": 10},
        }
        events = [encoder, gps_event(1.0)]

        modified, count = inject_gps_bias(
            events, 1.0, 2.0, 0.5, -0.3
        )

        self.assertEqual(count, 1)
        self.assertEqual(modified[0], encoder)
        self.assertAlmostEqual(modified[1]["data"]["x_m"], 2.5)
        self.assertAlmostEqual(modified[1]["data"]["y_m"], 2.7)
        self.assertEqual(
            modified[1]["data"]["nominal_variance_m2"],
            0.0025,
        )

    def test_missing_measurement_remains_missing(self):
        event = gps_event(1.0, available=False)

        modified, count = inject_gps_bias(
            [event], 1.0, 2.0, 0.5, -0.3
        )

        self.assertEqual(count, 0)
        self.assertEqual(modified[0], event)

    def test_half_open_interval(self):
        events = [
            gps_event(time_s)
            for time_s in (0.5, 1.0, 1.5, 2.0)
        ]

        modified, count = inject_gps_bias(
            events, 1.0, 2.0, 0.5, 0.0
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            [event["data"]["x_m"] for event in modified],
            [2.0, 2.5, 2.5, 2.0],
        )

    def test_original_input_unchanged(self):
        events = [gps_event(1.0)]
        original = deepcopy(events)

        modified, _ = inject_gps_bias(
            events, 1.0, 2.0, 0.5, -0.3
        )
        modified[0]["data"]["x_m"] = 999.0

        self.assertEqual(events, original)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            inject_gps_bias([], 2.0, 1.0, 0.5, 0.0)

        with self.assertRaises(ValueError):
            inject_gps_bias([], 1.0, 2.0, math.nan, 0.0)


if __name__ == "__main__":
    unittest.main()