"""Hipotez tabanlı seçimin komut döngüsüne bağlanması."""

from functools import partial
import unittest

from estimation.hypothesis_motion import (
    HypothesisMotionAgent,
    select_hypothesis_motion,
)
from estimation.scan_alignment_guard import (
    ScanGuardConfig,
)
from experiments.exp_29_complete_window import drive_trial
from experiments.exp_31_guarded_motion_benchmark import (
    check_pairing,
)
from experiments.exp_33_hypothesis_motion import (
    make_agent,
    raw_window,
)


class TestHypothesisMotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ScanGuardConfig(
            1.0,
            0.014248629353695821,
        )
        cls.factory = partial(
            make_agent,
            config=cls.config,
        )

        cls.adaptive = drive_trial(
            "adaptive",
            "straight",
            2.0,
            501,
            agent_factory=cls.factory,
        )
        cls.passive = drive_trial(
            "passive",
            "straight",
            2.0,
            501,
            agent_factory=cls.factory,
        )

    def test_healthy_does_not_request_maneuver(self):
        agent, _, _, _, checks, _ = drive_trial(
            "adaptive",
            "arc",
            0.0,
            501,
            agent_factory=self.factory,
        )

        self.assertFalse(agent.hold.active)
        self.assertIsNone(agent.decision)
        self.assertTrue(
            checks["required_horizon_completed"]
        )

    def test_first_alarm_prefix_matches_passive(self):
        checks = check_pairing({
            "adaptive": self.adaptive,
            "passive": self.passive,
        })
        self.assertTrue(all(checks.values()))

    def test_decision_contains_models_from_measurements(self):
        agent = self.adaptive[0]

        self.assertEqual(
            len(agent.decision["models"]), 2
        )
        self.assertEqual(
            len(agent.decision["candidates"]), 2
        )
        self.assertIn(
            agent.selected_motion,
            ("straight", "arc"),
        )
        self.assertNotIn(
            "fault_yaw_deg",
            agent.decision,
        )

    def test_selected_motion_is_applied_for_twenty_intervals(self):
        agent, _, commands, _, checks, _ = (
            self.adaptive
        )

        expected = {
            "straight": (4.0, 4.0),
            "arc": (3.1, 4.9),
        }[agent.selected_motion]

        intervals = [
            row
            for row in commands
            if (
                agent.hold.trigger_time_s - 1e-9
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

        self.assertTrue(
            checks["diagnostic_window_completed"]
        )

    def test_no_hypothesis_support_has_no_selected_motion(self):
        agent = self.adaptive[0]

        result = select_hypothesis_motion(
            agent.estimator,
            {"status": "insufficient_support"},
            agent.mounts,
        )

        self.assertIsNone(result["selected_motion"])

    def test_identical_predictions_do_not_force_motion(self):
        agent = self.adaptive[0]

        raw = {
            "status": "evaluated",
            "models": [
                {
                    "hypothesis": "front_yaw",
                    "best_offset_deg": 0.0,
                },
                {
                    "hypothesis": "rear_yaw",
                    "best_offset_deg": 0.0,
                },
            ],
        }

        result = select_hypothesis_motion(
            agent.estimator,
            raw,
            agent.mounts,
        )

        self.assertIsNone(result["selected_motion"])

    def test_prediction_does_not_mutate_estimator(self):
        agent = self.adaptive[0]
        state = agent.estimator.state.copy()
        covariance = agent.estimator.P.copy()

        raw = next(
            row
            for row in agent.raw_history
            if row["status"] == "evaluated"
        )

        select_hypothesis_motion(
            agent.estimator,
            raw,
            agent.mounts,
        )

        self.assertTrue(
            (state == agent.estimator.state).all()
        )
        self.assertTrue(
            (covariance == agent.estimator.P).all()
        )

    def test_raw_window_excludes_trigger_frame(self):
        agent = self.adaptive[0]

        window = raw_window(
            agent.raw_history,
            agent.hold.trigger_time_s,
            agent.window_end_s,
        )

        self.assertEqual(window["window_frames"], 4)
        self.assertEqual(
            window["supported_frames"], 4
        )

        self.assertIsNone(
            raw_window(
                agent.raw_history,
                None,
                None,
            )["mean_front_margin_m2"]
        )

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            HypothesisMotionAgent(
                "unknown",
                "arc",
                self.config,
            )


if __name__ == "__main__":
    unittest.main()