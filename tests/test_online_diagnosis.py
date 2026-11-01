import unittest

from estimation.diagnostic_motion import CANDIDATES
from estimation.online_diagnostic_agent import OnlineDiagnosticAgent
from experiments.exp_24_motion_comparison import generate_scenario
from experiments.exp_27_online_diagnosis import drive_case
from simulator.streaming_world import StreamingWorld


class TestOnlineDiagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.healthy = drive_case(0.)
        cls.mild = drive_case(2.)
        cls.strong = drive_case(5.)
        cls.fixed = drive_case(5., strategy='straight')

    def test_healthy_has_no_diagnostic_maneuver(self):
        agent, events, commands, _ = self.healthy
        self.assertFalse(agent.hold.active)
        self.assertIsNone(agent.decision)
        self.assertTrue(all(not row['differs_from_normal_route'] for row in commands))
        self.assertEqual(events, generate_scenario('arc', 0.)['events'])

    def test_mild_fault_is_not_forced_into_an_alarm(self):
        agent, _, _, _ = self.mild
        # Önceki EXP-24'te bu kayıt eşiği aşmamıştı. Kaçırma saklanmamalı.
        self.assertFalse(agent.hold.active)
        self.assertIsNone(agent.selected_motion)

    def test_strong_fault_triggers_before_maneuver_commands(self):
        agent, _, commands, _ = self.strong
        self.assertTrue(agent.hold.active)
        self.assertAlmostEqual(agent.hold.trigger_time_s, 2.)
        self.assertIn(agent.selected_motion, CANDIDATES)
        first_after = next(row for row in commands if row['interval_start_s'] >= 2. - 1e-9)
        self.assertEqual(
            (first_after['left_rate_rad_s'], first_after['right_rate_rad_s']),
            CANDIDATES[agent.selected_motion],
        )
        frozen_updates = [row['line_updates'] for row in agent.history if row['time_s'] >= 2.]
        self.assertEqual(len(set(frozen_updates)), 1)

    def test_fixed_strategy_changes_world_commands_after_alarm(self):
        agent, _, commands, _ = self.fixed
        self.assertEqual(agent.selected_motion, 'straight')
        self.assertTrue(all(
            row['left_rate_rad_s'] == 3.1 and row['right_rate_rad_s'] == 4.9
            for row in commands if row['interval_start_s'] < 2. - 1e-9
        ))
        first_after = next(row for row in commands if row['interval_start_s'] >= 2. - 1e-9)
        self.assertEqual((first_after['left_rate_rad_s'], first_after['right_rate_rad_s']), (4., 4.))

    def test_duplicate_batch_is_rejected(self):
        world = StreamingWorld()
        batch = world.initial_measurements()
        agent = OnlineDiagnosticAgent()
        agent.process(batch)
        with self.assertRaises(ValueError):
            agent.process(batch)
        with self.assertRaises(ValueError):
            world.initial_measurements()


if __name__ == '__main__':
    unittest.main()
