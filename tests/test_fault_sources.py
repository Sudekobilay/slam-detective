"""Hata izolasyonu, sabit referans ve ardışık karar testleri."""

import unittest

from estimation.scan_alignment_guard import ScanGuardConfig
from experiments.diagnosis_validation_common import (
    fault_recording,
    learn_reference,
    inspect_frozen,
    confirm_sequence,
)


class TestFaultSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events, cls.records = fault_recording(
            "straight",
            601,
            "healthy",
            0.0,
        )

        cls.reference = learn_reference(cls.events)
        cls.config = ScanGuardConfig(1.0, 0.02)

    def assert_isolated(self, source, target):
        events, records = fault_recording(
            "straight",
            601,
            source,
            5.0,
        )

        self.assertEqual(
            records["truth"],
            self.records["truth"],
        )

        changed = []

        for event, nominal in zip(
            events,
            self.events,
        ):
            if event != nominal:
                changed.append(event)

                self.assertEqual(
                    event["sensor_id"],
                    target,
                )
                self.assertGreaterEqual(
                    event["time_s"],
                    2.0,
                )

        self.assertEqual(len(changed), 5)

    def test_front_fault_changes_only_front_after_start(self):
        self.assert_isolated(
            "front_yaw",
            "front",
        )

    def test_rear_fault_changes_only_rear_after_start(self):
        self.assert_isolated(
            "rear_yaw",
            "rear",
        )

    def test_heading_case_does_not_change_sensor_recording(self):
        events, records = fault_recording(
            "straight",
            601,
            "heading_offset",
            5.0,
        )

        self.assertEqual(events, self.events)
        self.assertEqual(
            records["truth"],
            self.records["truth"],
        )

    def test_reference_state_and_covariance_remain_unchanged(self):
        state = self.reference.state.copy()
        covariance = self.reference.P.copy()

        frames = inspect_frozen(
            self.events,
            self.reference,
            self.config,
        )

        self.assertEqual(len(frames), 5)
        self.assertTrue(
            (self.reference.state == state).all()
        )
        self.assertTrue(
            (self.reference.P == covariance).all()
        )

    def test_heading_model_estimates_opposite_correction(self):
        frames = inspect_frozen(
            self.events,
            self.reference,
            self.config,
            heading_perturbation_deg=5.0,
        )

        best = frames[0]["raw"]["models"][0]

        self.assertEqual(
            best["hypothesis"],
            "heading_offset",
        )
        self.assertAlmostEqual(
            best["best_offset_deg"],
            -5.0,
            delta=0.5,
        )

    def test_confirmation_needs_two_same_sign_consecutive_candidates(self):
        def frame(time, angle):
            return {
                "time_s": time,
                "decision": {
                    "status": "candidate",
                    "hypothesis": "front_yaw",
                    "offset_deg": angle,
                },
            }

        self.assertEqual(
            confirm_sequence([
                frame(2.0, 2.0),
            ])["status"],
            "uncertain",
        )

        result = confirm_sequence([
            frame(2.0, 2.0),
            frame(2.5, 2.0),
        ])

        self.assertEqual(
            result["hypothesis"],
            "front_yaw",
        )
        self.assertEqual(
            result["confirmation_time_s"],
            2.5,
        )

        self.assertEqual(
            confirm_sequence([
                frame(2.0, 2.0),
                frame(2.5, -2.0),
            ])["status"],
            "uncertain",
        )

    def test_uncertain_frame_breaks_consecutive_confirmation(self):
        candidate = {
            "status": "candidate",
            "hypothesis": "front_yaw",
            "offset_deg": 2.0,
        }

        frames = [
            {
                "time_s": 2.0,
                "decision": candidate,
            },
            {
                "time_s": 2.5,
                "decision": {
                    "status": "uncertain",
                },
            },
            {
                "time_s": 3.0,
                "decision": candidate,
            },
        ]

        self.assertEqual(
            confirm_sequence(frames)["status"],
            "uncertain",
        )

    def test_invalid_source_or_angle_rejected(self):
        for source, angle in (
            ("unknown", 2.0),
            ("healthy", 2.0),
            ("rear_yaw", float("nan")),
        ):
            with self.assertRaises(ValueError):
                fault_recording(
                    "straight",
                    601,
                    source,
                    angle,
                )


if __name__ == "__main__":
    unittest.main()