"""Ek izleyicinin alarm, harita koruması ve eski davranış sözleşmeleri."""
import copy
import math
from types import SimpleNamespace
import unittest

from estimation.scan_alignment_guard import (
    CombinedLidarGate, ScanAlignmentGuard, ScanGuardConfig, scan_alignment,
)
from experiments.exp_30_scan_guard_audit import fixed_recording, replay_monitor
from simulator.mounting import SensorMount
from simulator.robot import Pose


def alignment(offset, rms=.001, edge=False):
    return {'offset_deg': offset, 'alignment_rms_m': rms, 'at_search_edge': edge}


def pair(front, rear=0.):
    return {'front': alignment(front), 'rear': alignment(rear)}


class TestScanAlignmentGuard(unittest.TestCase):
    def setUp(self):
        self.config = ScanGuardConfig(1., .02)
        self.guard = ScanAlignmentGuard(self.config)

    def test_config_rejects_invalid_values(self):
        for kwargs in (
            {'relative_yaw_threshold_deg': float('nan')},
            {'max_alignment_rms_m': 0}, {'angle_step_deg': 0},
            {'trim_fraction': 1.1}, {'min_points': 2},
            {'consecutive_scans': 1},
        ):
            values = {'relative_yaw_threshold_deg': 1., 'max_alignment_rms_m': .02}
            values.update(kwargs)
            with self.assertRaises(ValueError):
                ScanGuardConfig(**values)

    def test_common_heading_offset_does_not_trigger(self):
        for time in (0., .5, 1.):
            self.guard.update_alignment_result(time, pair(2., 2.))
        self.assertFalse(self.guard.confirmed)
        self.assertFalse(self.guard.quarantine_active)

    def test_warning_protects_before_confirmation(self):
        first = self.guard.update_alignment_result(2., pair(2.))
        self.assertEqual(first['status'], 'watch')
        self.assertTrue(self.guard.quarantine_active)
        self.assertFalse(self.guard.confirmed)
        self.guard.update_alignment_result(2.5, pair(2.))
        self.assertTrue(self.guard.confirmed)

    def test_opposite_sign_restarts_confirmation(self):
        self.guard.update_alignment_result(2., pair(2.))
        result = self.guard.update_alignment_result(2.5, pair(-2.))
        self.assertEqual(result['consecutive_scans'], 1)
        self.assertFalse(self.guard.confirmed)

    def test_insufficient_support_preserves_protection_resets_streak(self):
        self.guard.update_alignment_result(2., pair(2.))
        self.guard.update_alignment_result(2.5, {'front': None, 'rear': alignment(0.)})
        self.assertTrue(self.guard.quarantine_active)
        self.guard.update_alignment_result(3., pair(2.))
        self.assertFalse(self.guard.confirmed)

    def test_healthy_frame_clears_unconfirmed_warning(self):
        self.guard.update_alignment_result(2., pair(2.))
        self.guard.update_alignment_result(2.5, pair(.5))
        self.assertFalse(self.guard.quarantine_active)

    def test_bad_fit_or_search_edge_is_not_confirmation(self):
        for time, result in ((0., alignment(2., rms=.03)), (.5, alignment(2., edge=True))):
            frame = self.guard.update_alignment_result(time, {'front': result, 'rear': alignment(0.)})
            self.assertEqual(frame['status'], 'insufficient_alignment_support')
        self.assertFalse(self.guard.confirmed)

    def test_invalid_input_does_not_advance_clock(self):
        with self.assertRaises(ValueError):
            self.guard.update_alignment_result(2., pair(float('nan')))
        self.guard.update_alignment_result(2., pair(0.))
        with self.assertRaises(ValueError):
            self.guard.update_alignment_result(2., pair(0.))

    def test_combined_gate_preserves_nis_fast_path(self):
        gate = CombinedLidarGate(9.21034, self.guard)
        evidence = [{'sensor_id': 'front', 'observation_index': 0, 'nis': 10.}]
        self.assertTrue(gate.observe(2., evidence))
        self.assertEqual(gate.trigger_source, 'matched_nis')
        self.assertEqual(gate.trigger_evidence[0]['nis'], 10.)

    def test_consensus_evidence_is_not_fabricated_nis(self):
        gate = CombinedLidarGate(9.21034, self.guard)
        self.guard.update_alignment_result(2., pair(2.))
        self.assertFalse(gate.observe(2., []))
        self.guard.update_alignment_result(2.5, pair(2.))
        self.assertTrue(gate.observe(2.5, []))
        self.assertEqual(gate.trigger_source, 'scan_alignment_consensus')
        self.assertNotIn('nis', gate.trigger_evidence[0])

    def test_raw_alignment_recovers_rotation_without_mutation(self):
        # Analytic wall fixture: horizontal and vertical segments, scan points
        # expressed in a sensor whose mounting yaw is +5 degrees.
        features = (
            SimpleNamespace(start=(2., -2.), end=(2., 2.)),
            SimpleNamespace(start=(-2., 2.), end=(2., 2.)),
        )
        world = [(2., -1.5 + i * .1) for i in range(30)]
        world += [(-1.5 + i * .1, 2.) for i in range(30)]
        angle = math.radians(-5.)
        c, s = math.cos(angle), math.sin(angle)
        points = [[c*x-s*y, s*x+c*y] for x, y in world]
        packet = {'data': {'scan': {'points': points}}}
        before = copy.deepcopy(packet)
        estimator = SimpleNamespace(pose=Pose(0., 0., 0.), map_features=lambda: features)
        result = scan_alignment(estimator, packet, SensorMount(), self.config)
        self.assertEqual(result['offset_deg'], 5.)
        self.assertLess(result['alignment_rms_m'], 1e-12)
        self.assertEqual(packet, before)
        self.assertEqual(estimator.pose, Pose(0., 0., 0.))

    def test_no_map_returns_insufficient_support(self):
        estimator = SimpleNamespace(map_features=lambda: ())
        self.assertIsNone(scan_alignment(estimator, {'data': {'scan': {'points': []}}}, SensorMount(), self.config))

    def test_known_missed_fault_detected_without_changing_input(self):
        events, _ = fixed_recording('straight', 303, 5.)
        before = copy.deepcopy(events)
        original = replay_monitor(events, 'straight')
        guarded = replay_monitor(events, 'straight', self.config)
        self.assertFalse(original.hold.active)
        self.assertEqual(guarded.hold.trigger_time_s, 2.5)
        self.assertEqual(guarded.hold.nis_threshold, 9.21034)
        self.assertEqual(events, before)

    def test_small_fault_confirmation_and_healthy_control(self):
        healthy, _ = fixed_recording('arc', 101, 0.)
        faulty, _ = fixed_recording('arc', 101, 2.)
        self.assertFalse(replay_monitor(healthy, 'arc', self.config).hold.active)
        self.assertEqual(replay_monitor(faulty, 'arc', self.config).hold.trigger_time_s, 2.5)


if __name__ == '__main__':
    unittest.main()
