"""GPS izleyicisi ve kayıt izolasyonu testleri."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from estimation.gps_consistency import (
    GPSConsistencyMonitor,
    joint_status,
)
from estimation.scan_alignment_guard import (
    ScanGuardConfig,
)
from experiments.diagnosis_validation_common import (
    fault_recording,
)
from experiments.exp_35_joint_gps import (
    add_gps_bias,
    replay_joint,
)
from simulator.gps import GPSMeasurement


def reference(time):
    return SimpleNamespace(
        time_s=time,
        state=np.array([1.0, 3.0, 0.0, 0.0]),
        P=np.eye(4) * 0.001,
    )


def packet(
    time,
    x=1.0,
    y=3.0,
    available=True,
):
    return GPSMeasurement(
        time,
        available,
        x if available else None,
        y if available else None,
        0.0025,
    )


class TestGPSConsistency(unittest.TestCase):
    def test_nominal_measurement_has_zero_nis(self):
        monitor = GPSConsistencyMonitor()

        result = monitor.observe(
            reference(0.0),
            packet(0.0),
        )

        self.assertEqual(result["nis"], 0.0)
        self.assertFalse(monitor.confirmed)

    def test_three_exceedances_confirm_after_instant_rejection(self):
        monitor = GPSConsistencyMonitor()

        for index, time in enumerate(
            (3.0, 3.2, 3.4)
        ):
            row = monitor.observe(
                reference(time),
                packet(time, 1.5, 3.3),
            )

            self.assertTrue(row["instant_reject"])
            self.assertEqual(
                monitor.confirmed,
                index == 2,
            )

        self.assertEqual(
            monitor.confirmation_time_s,
            3.4,
        )

    def test_missing_measurement_resets_streak_not_treated_as_position(self):
        monitor = GPSConsistencyMonitor()

        monitor.observe(
            reference(0.0),
            packet(0.0, 1.5, 3.3),
        )

        result = monitor.observe(
            reference(0.2),
            packet(0.2, available=False),
        )

        self.assertEqual(monitor.streak, 0)
        self.assertIsNone(result["nis"])
        self.assertFalse(result["instant_reject"])

    def test_reference_state_and_covariance_not_mutated(self):
        estimator = reference(0.0)
        before = copy.deepcopy(estimator)

        GPSConsistencyMonitor().observe(
            estimator,
            packet(0.0, 1.5, 3.3),
        )

        self.assertTrue(
            (
                estimator.state == before.state
            ).all()
        )
        self.assertTrue(
            (
                estimator.P == before.P
            ).all()
        )

    def test_invalid_time_and_options_rejected(self):
        for threshold, consecutive in (
            (0.0, 3),
            (9.0, 1),
            (float("nan"), 3),
        ):
            with self.assertRaises(ValueError):
                GPSConsistencyMonitor(
                    threshold,
                    consecutive,
                )

        monitor = GPSConsistencyMonitor()
        monitor.observe(
            reference(0.0),
            packet(0.0),
        )

        with self.assertRaises(ValueError):
            monitor.observe(
                reference(0.0),
                packet(0.0),
            )

    def test_joint_status_can_keep_two_suspicions(self):
        self.assertEqual(
            joint_status(True, True),
            "multiple_inconsistencies",
        )
        self.assertEqual(
            joint_status(False, True),
            "gps_reference_inconsistency",
        )
        self.assertEqual(
            joint_status(True, False),
            "geometric_inconsistency",
        )
        self.assertEqual(
            joint_status(False, False),
            "no_confirmed_inconsistency",
        )

    def test_bias_changes_only_available_gps_after_start(self):
        events, _ = fault_recording(
            "straight",
            701,
            "healthy",
            0.0,
        )

        before = copy.deepcopy(events)
        biased = add_gps_bias(events, 0.5, 0.3)

        changes = [
            first
            for first, second in zip(
                biased,
                events,
            )
            if first != second
        ]

        self.assertEqual(len(changes), 6)

        self.assertTrue(all(
            event["sensor_type"] == "gps"
            and event["time_s"] >= 3.0
            for event in changes
        ))

        self.assertEqual(events, before)

    def test_reference_is_independent_of_gps_bias_in_full_replay(self):
        events, _ = fault_recording(
            "straight",
            701,
            "healthy",
            0.0,
        )

        config = ScanGuardConfig(
            1.0,
            0.014248629353695821,
        )

        healthy = replay_joint(events, config)
        faulty = replay_joint(
            add_gps_bias(events, 0.5, 0.3),
            config,
        )

        self.assertTrue(
            (
                healthy[0].estimator.state
                == faulty[0].estimator.state
            ).all()
        )

        self.assertTrue(faulty[2].confirmed)

        self.assertEqual(
            faulty[2].confirmation_time_s,
            3.4,
        )
        self.assertEqual(
            faulty[-1],
            "gps_reference_inconsistency",
        )
        self.assertGreater(
            faulty[3]["guarded_gps"],
            0,
        )
        self.assertGreater(faulty[4], 0)


if __name__ == "__main__":
    unittest.main()