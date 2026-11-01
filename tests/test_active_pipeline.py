"""Bütünleşik denetim ve GPS izolasyonu testleri."""

from itertools import groupby
import unittest

import numpy as np

from estimation.active_pipeline import (
    ActivePipeline,
    make_backend,
    geometry_view,
    gps_reference,
)
from estimation.diagnostic_motion import CANDIDATES
from estimation.scan_alignment_guard import ScanGuardConfig
from experiments.exp_30_scan_guard_audit import (
    fixed_recording,
)
from experiments.exp_39_active_pipeline import (
    run_active,
)


class TestActivePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ScanGuardConfig(
            1.0,
            0.014248629353695821,
        )

        cls.adaptive = run_active(
            "ekf",
            cls.config,
            use_gps=True,
        )
        cls.passive = run_active(
            "ekf",
            cls.config,
            policy="passive",
            use_gps=True,
        )

    def test_backend_factory_covers_three_algorithms(self):
        for name in (
            "ekf", "fastslam1", "fastslam2"
        ):
            self.assertTrue(callable(
                make_backend(name, 4).snapshot
            ))

        with self.assertRaises(ValueError):
            make_backend("unknown")

    def test_particle_geometry_uses_same_particle_pose_and_map(self):
        backend = make_backend("fastslam1", 4)
        backend.log_weights[:] = -100.0
        backend.log_weights[2] = 0.0

        backend.particles[2].state[:3] = [
            5.0, 6.0, 0.2
        ]

        view = geometry_view(backend)

        self.assertEqual(view.pose.x, 5.0)
        self.assertEqual(view.pose.y, 6.0)
        self.assertAlmostEqual(view.pose.theta, 0.2)

    def test_particle_gps_reference_has_finite_covariance(self):
        reference = gps_reference(
            make_backend("fastslam2", 8)
        )

        self.assertEqual(
            reference.P.shape,
            (4, 4),
        )
        self.assertTrue(
            np.isfinite(reference.P).all()
        )
        self.assertGreater(
            np.linalg.eigvalsh(reference.P).min(),
            0.0,
        )

    def test_alarm_prefix_is_equal_between_policies(self):
        first, events1, *_ = self.adaptive
        second, events2, *_ = self.passive

        self.assertEqual(
            first.trigger_time_s,
            second.trigger_time_s,
        )
        self.assertEqual(
            [
                event
                for event in events1
                if event["time_s"]
                <= first.trigger_time_s
            ],
            [
                event
                for event in events2
                if event["time_s"]
                <= second.trigger_time_s
            ],
        )

    def test_selected_motion_is_applied_for_complete_window(self):
        agent, _, commands, _, reason = (
            self.adaptive
        )

        self.assertIsNone(reason)
        self.assertEqual(
            agent.trigger_time_s,
            2.5,
        )

        expected = CANDIDATES[
            agent.selected_motion
        ]

        intervals = [
            row
            for row in commands
            if (
                agent.trigger_time_s - 1e-9
                <= row["interval_start_s"]
                < agent.window_end_s - 1e-9
            )
        ]

        self.assertEqual(len(intervals), 20)
        self.assertTrue(all(
            (
                row["left_rate_rad_s"],
                row["right_rate_rad_s"],
            ) == expected
            for row in intervals
        ))
        self.assertAlmostEqual(
            agent.time_s,
            agent.window_end_s,
        )

    def test_gps_only_trigger_can_decline_unsupported_motion(self):
        agent, _, commands, _, reason = run_active(
            "ekf",
            self.config,
            source="gps",
            use_gps=True,
        )

        self.assertIsNone(reason)
        self.assertTrue(
            agent.gps_monitor.confirmed
        )
        self.assertFalse(agent.guard.confirmed)
        self.assertIsNone(agent.selected_motion)
        self.assertEqual(
            agent.trigger_source,
            "gps_reference",
        )

        self.assertTrue(any(
            row["mode"] == "NO_SUPPORTED_MANEUVER"
            and row["left_rate_rad_s"]
            == row["right_rate_rad_s"]
            == 0.0
            for row in commands
        ))

    def test_combined_case_retains_both_suspicions(self):
        agent, _, _, _, reason = run_active(
            "ekf",
            self.config,
            source="combined",
            use_gps=True,
        )

        self.assertIsNone(reason)
        self.assertEqual(
            agent.history[-1]["joint_status"],
            "multiple_inconsistencies",
        )
        self.assertGreater(
            agent.gps_rejected,
            0,
        )

    def test_healthy_case_has_no_forced_maneuver(self):
        agent, _, _, _, reason = run_active(
            "ekf",
            self.config,
            source="healthy",
            use_gps=True,
        )

        self.assertIsNone(reason)
        self.assertIsNone(
            agent.trigger_time_s
        )
        self.assertAlmostEqual(
            agent.time_s,
            4.0,
        )

    def test_duplicate_time_rejected(self):
        agent = ActivePipeline(
            "ekf",
            self.config,
        )

        events, _ = fixed_recording(
            "straight",
            1001,
            0.0,
        )

        batch = [
            event
            for event in events
            if event["time_s"] == 0.0
        ]

        agent.process(batch)

        with self.assertRaises(ValueError):
            agent.process(batch)

    def test_gps_fusion_does_not_change_control_reference_on_same_record(self):
        first = ActivePipeline(
            "ekf",
            self.config,
            use_gps=False,
        )
        second = ActivePipeline(
            "ekf",
            self.config,
            use_gps=True,
        )

        events, _ = fixed_recording(
            "straight",
            1001,
            0.0,
        )

        for _, batch in groupby(
            events,
            key=lambda event: event["time_s"],
        ):
            batch = list(batch)
            first.process(batch)
            second.process(batch)

        self.assertEqual(
            first.reference_path,
            second.reference_path,
        )
        self.assertEqual(
            first.raw_history,
            second.raw_history,
        )


if __name__ == "__main__":
    unittest.main()