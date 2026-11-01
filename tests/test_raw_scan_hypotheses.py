"""Ham hipotez bankasının geometrisi ve belirsiz kararları."""

import copy
import math
from types import SimpleNamespace
import unittest

import numpy as np

from estimation.raw_scan_hypotheses import (
    compare_raw_hypotheses,
    diagnose_raw,
)
from estimation.scan_alignment_guard import (
    ScanGuardConfig,
)
from simulator.mounting import SensorMount
from simulator.robot import Pose


class TestRawScanHypotheses(unittest.TestCase):
    def setUp(self):
        self.config = ScanGuardConfig(1.0, 0.02)

        self.mounts = {
            "front": SensorMount(),
            "rear": SensorMount(),
        }

        self.features = (
            SimpleNamespace(
                start=(2.0, -2.0),
                end=(2.0, 2.0),
            ),
            SimpleNamespace(
                start=(-2.0, 2.0),
                end=(2.0, 2.0),
            ),
        )

        self.estimator = SimpleNamespace(
            pose=Pose(0.0, 0.0, 0.0),
            map_features=lambda: self.features,
        )

        self.world = [
            (2.0, -1.5 + index * 0.1)
            for index in range(30)
        ] + [
            (-1.5 + index * 0.1, 2.0)
            for index in range(30)
        ]

        self.thresholds = {
            "minimum_improvement_m2": 1e-6,
            "minimum_model_gap_m2": 1e-6,
        }

    def packets(self, front=0.0, rear=0.0):
        packets = []

        for sensor, angle in (
            ("front", front),
            ("rear", rear),
        ):
            cosine = math.cos(math.radians(-angle))
            sine = math.sin(math.radians(-angle))

            points = [
                [
                    cosine * x - sine * y,
                    sine * x + cosine * y,
                ]
                for x, y in self.world
            ]

            packets.append({
                "sensor_id": sensor,
                "data": {
                    "scan": {
                        "points": points,
                    },
                },
            })

        return packets

    def result(self, front=0.0, rear=0.0):
        return compare_raw_hypotheses(
            self.estimator,
            self.packets(front, rear),
            self.mounts,
            self.config,
        )

    def test_front_rotation_is_candidate(self):
        result = self.result(5.0, 0.0)
        best = result["models"][0]

        self.assertEqual(
            best["hypothesis"],
            "front_yaw",
        )
        self.assertEqual(
            best["best_offset_deg"],
            5.0,
        )
        self.assertLess(
            best["cost_m2"],
            1e-20,
        )

        decision = diagnose_raw(
            result,
            self.thresholds,
            0.02,
        )
        self.assertEqual(
            decision["hypothesis"],
            "front_yaw",
        )

    def test_rear_rotation_is_candidate(self):
        result = self.result(0.0, -5.0)

        self.assertEqual(
            result["models"][0]["hypothesis"],
            "rear_yaw",
        )

    def test_common_rotation_prefers_heading_model(self):
        result = self.result(3.0, 3.0)

        self.assertEqual(
            result["models"][0]["hypothesis"],
            "heading_offset",
        )
        self.assertEqual(
            result["models"][0]["best_offset_deg"],
            3.0,
        )

    def test_nominal_does_not_invent_fault(self):
        decision = diagnose_raw(
            self.result(),
            self.thresholds,
            0.02,
        )
        self.assertEqual(
            decision["status"],
            "nominal_consistent",
        )

    def test_missing_sensor_stays_uncertain(self):
        result = compare_raw_hypotheses(
            self.estimator,
            self.packets()[:1],
            self.mounts,
            self.config,
        )

        decision = diagnose_raw(
            result,
            self.thresholds,
            0.02,
        )
        self.assertEqual(
            decision["reason"],
            "insufficient_support",
        )

    def test_inputs_and_pose_not_mutated(self):
        packets = self.packets(5.0)
        before = copy.deepcopy(packets)

        compare_raw_hypotheses(
            self.estimator,
            packets,
            self.mounts,
            self.config,
        )

        self.assertEqual(packets, before)
        self.assertEqual(
            self.estimator.pose,
            Pose(0.0, 0.0, 0.0),
        )

    def test_duplicate_sensor_rejected(self):
        packets = self.packets()

        with self.assertRaises(ValueError):
            compare_raw_hypotheses(
                self.estimator,
                [packets[0], packets[0]],
                self.mounts,
                self.config,
            )

    def test_boundary_fit_is_uncertain(self):
        decision = diagnose_raw(
            self.result(10.0),
            self.thresholds,
            0.02,
        )
        self.assertEqual(
            decision["status"],
            "uncertain",
        )

    def test_close_competitors_are_not_forced_to_a_label(self):
        result = self.result(5.0)

        result["models"][1]["cost_m2"] = (
            result["models"][0]["cost_m2"]
        )

        decision = diagnose_raw(
            result,
            self.thresholds,
            0.02,
        )
        self.assertEqual(
            decision["reason"],
            "competing_models_close",
        )

    def test_nonfinite_points_rejected(self):
        packets = self.packets()
        packets[0]["data"]["scan"]["points"][0][0] = (
            np.nan
        )

        with self.assertRaises(ValueError):
            compare_raw_hypotheses(
                self.estimator,
                packets,
                self.mounts,
                self.config,
            )


if __name__ == "__main__":
    unittest.main()