"""Sensör yapılandırması ve ortak SLAM arayüzü testleri."""

import copy
import unittest

from configuration.layouts import examples, read_layout
from experiments.exp_36_sensor_layouts import make_recording, replay_layout


class TestSensorLayouts(unittest.TestCase):
    def test_example_sensor_counts(self):
        for name, count in (
            ("single", 1),
            ("dual", 2),
            ("quad", 4),
        ):
            self.assertEqual(
                len(read_layout(examples()[name])),
                count,
            )

    def test_duplicate_and_reserved_ids_rejected(self):
        for value in ("front", "gps"):
            data = examples()["dual"]
            data["lidars"][1]["id"] = value

            with self.assertRaises(ValueError):
                read_layout(data)

    def test_invalid_geometry_and_unknown_fields_rejected(self):
        data = examples()["single"]
        data["lidars"][0]["yaw_deg"] = float("nan")

        with self.assertRaises(ValueError):
            read_layout(data)

        data = examples()["single"]
        data["unsupported"] = True

        with self.assertRaises(ValueError):
            read_layout(data)

    def test_mount_position_and_yaw_follow_config(self):
        data = examples()["single"]

        data["lidars"][0].update(
            x_m=0.1,
            y_m=-0.2,
            yaw_deg=90.0,
        )

        unit = read_layout(data)[0]

        self.assertEqual(
            unit.actual_mount.x,
            0.1,
        )
        self.assertEqual(
            unit.actual_mount.y,
            -0.2,
        )
        self.assertAlmostEqual(
            unit.actual_mount.yaw,
            1.5707963267948966,
        )

    def test_layout_change_preserves_other_sensors_and_truth(self):
        first, truth1 = make_recording(
            examples()["single"]
        )
        second, truth2 = make_recording(
            examples()["quad"]
        )

        self.assertEqual(
            truth1["truth"],
            truth2["truth"],
        )

        self.assertEqual(
            [
                event
                for event in first
                if event["sensor_type"] != "lidar"
            ],
            [
                event
                for event in second
                if event["sensor_type"] != "lidar"
            ],
        )

    def test_gps_can_be_disabled_without_changing_other_measurements(self):
        data = examples()["dual"]
        enabled, _ = make_recording(data)

        data["gps_enabled"] = False
        disabled, _ = make_recording(data)

        self.assertEqual(
            disabled,
            [
                event
                for event in enabled
                if event["sensor_type"] != "gps"
            ],
        )

        backend, path = replay_layout(
            disabled,
            data,
        )

        self.assertEqual(
            backend.estimator.gps_updates,
            0,
        )
        self.assertEqual(len(path), 41)

    def test_quad_replay_creates_map_without_mutating_input(self):
        data = examples()["quad"]
        events, _ = make_recording(data)
        before = copy.deepcopy(events)

        backend, path = replay_layout(
            events,
            data,
        )

        self.assertEqual(len(path), 41)
        self.assertGreater(
            backend.estimator.landmark_count,
            0,
        )
        self.assertEqual(events, before)

        self.assertEqual(
            set(backend.mounts),
            {"front", "rear", "left", "right"},
        )

    def test_unknown_scan_id_rejected(self):
        data = examples()["single"]
        events, _ = make_recording(data)

        event = next(
            event
            for event in events
            if event["sensor_type"] == "lidar"
        )
        event["sensor_id"] = "unknown"

        with self.assertRaises(ValueError):
            replay_layout(events, data)


if __name__ == "__main__":
    unittest.main()