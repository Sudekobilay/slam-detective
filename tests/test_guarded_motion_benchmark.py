"""EXP-31: politika ayrışması ve eşit pencere denetimleri."""

from functools import partial
import unittest

from estimation.scan_alignment_guard import ScanGuardConfig
from experiments.exp_29_complete_window import drive_trial
from experiments.exp_31_guarded_motion_benchmark import POLICIES, check_pairing, make_guarded_agent, summarize
from evaluation.movement import movement_metrics


class TestGuardedMotionBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factory = partial(
            make_guarded_agent,
            config=ScanGuardConfig(
                1.0,
                0.014248629353695821,
            ),
        )

        cls.trials = {
            policy: drive_trial(
                policy,
                "straight",
                2.0,
                303,
                agent_factory=cls.factory,
            )
            for policy in POLICIES
        }

    def test_same_real_sensor_history_through_first_alarm(self):
        self.assertTrue(
            all(check_pairing(self.trials).values())
        )
        self.assertEqual(
            self.trials["adaptive"][0].hold.trigger_time_s,
            2.5,
        )

    def test_all_policies_complete_two_seconds_after_confirmation(self):
        for trial in self.trials.values():
            agent, _, commands, _, checks, _ = trial

            self.assertTrue(
                checks["diagnostic_window_completed"]
            )
            self.assertTrue(
                checks["required_horizon_completed"]
            )
            self.assertAlmostEqual(
                agent.estimator.time_s,
                4.5,
            )

            intervals = sum(
                command["applied"]
                and 2.5 - 1e-9
                <= command["interval_start_s"]
                < 4.5 - 1e-9
                for command in commands
            )
            self.assertEqual(intervals, 20)

    def test_fixed_policies_apply_distinct_real_commands(self):
        straight = self.trials["fixed_straight"][2]
        arc = self.trials["fixed_arc"][2]

        straight_command = next(
            command
            for command in straight
            if abs(command["interval_start_s"] - 2.5) < 1e-9
        )
        arc_command = next(
            command
            for command in arc
            if abs(command["interval_start_s"] - 2.5) < 1e-9
        )

        self.assertEqual(
            (
                straight_command["left_rate_rad_s"],
                straight_command["right_rate_rad_s"],
            ),
            (4.0, 4.0),
        )
        self.assertEqual(
            (
                arc_command["left_rate_rad_s"],
                arc_command["right_rate_rad_s"],
            ),
            (3.1, 4.9),
        )
        self.assertNotEqual(
            self.trials["fixed_straight"][3]["truth"][-1],
            self.trials["fixed_arc"][3]["truth"][-1],
        )

    def test_passive_never_changes_commands(self):
        agent, _, commands, records, _, _ = (
            self.trials["passive"]
        )

        cost = movement_metrics(
            records["truth"],
            commands,
            agent.hold.trigger_time_s,
            agent.window_end_s,
        )

        self.assertAlmostEqual(
            cost["diagnostic_duration_s"], 2.0
        )
        self.assertAlmostEqual(
            cost["diagnostic_distance_m"],
            0.4,
            places=10,
        )
        self.assertEqual(
            cost["changed_command_duration_s"], 0.0
        )
        self.assertEqual(
            cost["diagnostic_absolute_turn_rad"], 0.0
        )

    def test_turning_cost_reports_actual_duration_and_yaw(self):
        agent, _, commands, records, _, _ = (
            self.trials["fixed_arc"]
        )

        cost = movement_metrics(
            records["truth"],
            commands,
            agent.hold.trigger_time_s,
            agent.window_end_s,
        )

        self.assertAlmostEqual(
            cost["changed_command_duration_s"], 2.0
        )
        self.assertAlmostEqual(
            cost["diagnostic_absolute_turn_rad"], 0.6
        )

    def test_healthy_has_no_artificial_diagnostic_window(self):
        agent, _, commands, records, checks, _ = drive_trial(
            "adaptive",
            "arc",
            0.0,
            401,
            agent_factory=self.factory,
        )

        self.assertFalse(agent.hold.active)
        self.assertAlmostEqual(
            agent.estimator.time_s, 4.0
        )
        self.assertFalse(
            checks["diagnostic_window_completed"]
        )

        cost = movement_metrics(
            records["truth"],
            commands,
            None,
            None,
        )
        self.assertTrue(
            all(value is None for value in cost.values())
        )

    def test_factory_rejects_unknown_policy(self):
        with self.assertRaises(ValueError):
            self.factory("unknown", "arc")

    def test_pairing_detects_changed_prefix(self):
        agent, events, *rest = self.trials["passive"]

        changed = [dict(event) for event in events]
        changed[0] = {
            **changed[0],
            "changed": True,
        }

        bad = dict(self.trials)
        bad["fixed_arc"] = (agent, changed, *rest)

        self.assertFalse(
            check_pairing(bad)[
                "same_measurements_through_alarm"
            ]
        )

    def test_no_alarm_summary_does_not_invent_zero_diagnostic_metrics(self):
        row = {
            "normal_motion": "arc",
            "policy": "adaptive",
            "fault_yaw_deg": 0.0,
            "alarm": False,
            "diagnostic_window_completed": False,
            "required_horizon_completed": True,
            "premature_alarm": False,
            "detection_delay_s": None,
            "common_position_rmse_m": 0.001,
            "changed_command_duration_s": None,
            "selected_motion": None,
            "extra_simulation_time_s": 0.0,
        }

        group = summarize([row])[0]

        self.assertEqual(group["complete_windows"], 0)
        self.assertIsNone(group["pooled_front_margin"])
        self.assertIsNone(
            group["mean_diagnostic_distance_m"]
        )
        self.assertIsNone(
            group["mean_margin_difference_from_passive"]
        )


if __name__ == "__main__":
    unittest.main()