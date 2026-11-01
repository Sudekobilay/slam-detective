"""EXP-24'ün eşleştirilmiş senaryo ve hareket zamanlaması kontrolleri."""

import math
import unittest

from experiments.exp_24_motion_comparison import (
    BRANCH_STEP, BRANCH_TIME, MOTION_RATES, PREFIX_RATES,
    check_scenarios, generate_scenario, wheel_rates,
)


class TestMotionComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {
            f"{condition}_{motion}": generate_scenario(motion, error)
            for condition, error in (("healthy", 0.), ("front_fault", 5.))
            for motion in MOTION_RATES
        }

    def test_branch_starts_after_common_interval(self):
        for motion in MOTION_RATES:
            self.assertEqual(wheel_rates(motion, BRANCH_STEP), PREFIX_RATES)
            self.assertEqual(wheel_rates(motion, BRANCH_STEP + 1), MOTION_RATES[motion])
        with self.assertRaises(ValueError):
            wheel_rates("unknown", 1)
        with self.assertRaises(ValueError):
            wheel_rates("arc", 0)

    def test_same_history_before_motion_diverges(self):
        for condition in ("healthy", "front_fault"):
            a = self.cases[f"{condition}_straight"]
            b = self.cases[f"{condition}_arc"]
            self.assertEqual(a['truth'][:BRANCH_STEP + 1], b['truth'][:BRANCH_STEP + 1])
            self.assertEqual(
                [e for e in a['events'] if e['time_s'] <= BRANCH_TIME],
                [e for e in b['events'] if e['time_s'] <= BRANCH_TIME],
            )
            self.assertNotEqual(a['truth'][-1], b['truth'][-1])

    def test_only_front_measurements_change_with_fault(self):
        for motion in MOTION_RATES:
            healthy, faulty = self.cases[f"healthy_{motion}"], self.cases[f"front_fault_{motion}"]
            self.assertEqual(healthy['truth'], faulty['truth'])
            self.assertEqual(
                [e for e in healthy['events'] if e['sensor_id'] != 'front'],
                [e for e in faulty['events'] if e['sensor_id'] != 'front'],
            )
            self.assertEqual(len(faulty['metadata']['fault']['affected_scan_times_s']), 5)
            self.assertNotEqual(healthy['events'], faulty['events'])

    def test_equal_duration_and_traveled_distance(self):
        for case in self.cases.values():
            self.assertAlmostEqual(case['metadata']['duration_s'], 4.)
            self.assertAlmostEqual(case['metadata']['distance_m'], .8)
            for control in case['controls']:
                self.assertAlmostEqual(control['linear_speed_m_s'], .2)
        straight = self.cases['healthy_straight']['controls'][-1]
        arc = self.cases['healthy_arc']['controls'][-1]
        self.assertAlmostEqual(straight['yaw_rate_rad_s'], 0.)
        self.assertAlmostEqual(arc['yaw_rate_rad_s'], .3)

    def test_record_integrity_and_no_fault_label_in_events(self):
        checks = check_scenarios(self.cases)
        self.assertTrue(all(checks.values()), checks)
        for case in self.cases.values():
            self.assertEqual(len(case['events']), 119)
            for event in case['events']:
                self.assertEqual(set(event), {
                    'schema_version', 'time_s', 'sensor_type', 'sensor_id', 'data'
                })
                for forbidden in ('ground_truth', 'true_pose', 'actual_mount', 'fault', 'fault_label'):
                    self.assertNotIn(forbidden, event['data'])


if __name__ == '__main__':
    unittest.main()
