"""Yapılandırma, geçmiş deney uyumu ve dört LiDAR'da gerçek komut kontrolü."""
import copy
from itertools import groupby
import json
import unittest
from unittest.mock import patch

from estimation.active_pipeline import ActivePipeline
from estimation.configurable_geometry import compare_layout_hypotheses, LayoutScanGuard
from estimation.raw_scan_hypotheses import compare_raw_hypotheses
from estimation.scan_alignment_guard import ScanGuardConfig
from experiments.exp_30_scan_guard_audit import fixed_recording
from configuration.layouts import examples, read_layout
from simulator.layout_world import LayoutWorld
from application.layout_service import calibrate_layout, create_agent, run_layout
from tests import test_raw_scan_hypotheses as raw_fixture


class TestConfigurableActive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dual, cls.quad = examples()['dual'], examples()['quad']
        cls.config, _ = calibrate_layout(cls.quad)
        cls.active = run_layout(cls.quad, cls.config)
        cls.passive = run_layout(cls.quad, cls.config, policy='passive')

    def test_legacy_default_is_unchanged(self):
        old = ActivePipeline('ekf', ScanGuardConfig(1., .014248629353695821))
        new = create_agent(self.dual, old.config, use_gps=False)
        events, _ = fixed_recording('straight', 1001, 5.)
        for _, batch in groupby(events, key=lambda e: e['time_s']):
            batch = list(batch)
            old.process(batch)
            new.process(batch)
        self.assertEqual(old.reference_path, new.reference_path)
        self.assertEqual(old.trigger_time_s, new.trigger_time_s)
        self.assertEqual(old.selected_motion, new.selected_motion)

    def test_legacy_raw_model_costs_are_preserved(self):
        fixture = raw_fixture.TestRawScanHypotheses()
        fixture.setUp()
        args = (fixture.estimator, fixture.packets(5.), fixture.mounts, fixture.config)
        old = compare_raw_hypotheses(*args)
        new = compare_layout_hypotheses(*args)
        self.assertEqual(old['models'], new['models'])

    def test_arbitrary_sensor_names(self):
        data = copy.deepcopy(self.quad)
        for index, sensor in enumerate(data['lidars']):
            sensor['id'] = f'scanner_{index}'
        agent = create_agent(data, self.config, use_gps=False)
        world = LayoutWorld(data, 1001)
        agent.process(world.initial_measurements())
        agent.process(world.step(4., 4.))
        self.assertEqual(set(agent.mounts), {f'scanner_{i}' for i in range(4)})

    def test_quad_alarm_and_actual_motion(self):
        agent, _, commands, _, reason = self.active
        self.assertIsNone(reason)
        self.assertIsNotNone(agent.trigger_time_s)
        self.assertIn(agent.selected_motion, ('straight', 'arc'))
        selected = [r for r in commands if r['interval_start_s'] >= agent.trigger_time_s - 1e-9]
        self.assertEqual(len(selected), 20)
        self.assertTrue(all(r['applied'] for r in selected))
        json.dumps(agent.decision, allow_nan=False)
        for row in agent.decision['candidates']:
            self.assertEqual(set(row['common_ray_counts'][0]), set(agent.mounts))

    def test_quad_prefix_matches_passive(self):
        first, events1, *_ = self.active
        second, events2, *_ = self.passive
        self.assertEqual(first.trigger_time_s, second.trigger_time_s)
        self.assertEqual([e for e in events1 if e['time_s'] <= first.trigger_time_s],
                         [e for e in events2 if e['time_s'] <= second.trigger_time_s])

    def test_healthy_quad_has_no_alarm(self):
        agent, _, _, _, reason = run_layout(self.quad, self.config, fault='healthy')
        self.assertIsNone(reason)
        self.assertIsNone(agent.trigger_time_s)

    def test_single_lidar_is_explicitly_unsupported_for_isolation(self):
        data = examples()['single']
        agent, _, _, _, reason = run_layout(data, self.config)
        self.assertIsNone(reason)
        self.assertFalse(agent.guard.confirmed)
        self.assertTrue(all(r['status'] != 'evaluated' for r in agent.raw_history))
        self.assertIn('at_least_two_supported_lidars_required',
                      [r.get('reason') for r in agent.raw_history])

    def test_fault_changes_only_target_lidar_on_fixed_commands(self):
        healthy = LayoutWorld(self.quad, 1001)
        faulty = LayoutWorld(self.quad, 1001, 'lidar', 'left')
        self.assertEqual(healthy.initial_measurements(), faulty.initial_measurements())
        changed = False
        for _ in range(40):
            first, second = healthy.step(4., 4.), faulty.step(4., 4.)
            for a, b in zip(first, second):
                if a['sensor_id'] == 'left' and a['time_s'] >= 2.:
                    changed = changed or a != b
                else:
                    self.assertEqual(a, b)
        self.assertTrue(changed)
        self.assertEqual(healthy.evaluation_records()['truth'], faulty.evaluation_records()['truth'])

    def test_all_backends_receive_configured_mounts(self):
        for name in ('ekf', 'fastslam1', 'fastslam2'):
            agent = create_agent(self.quad, self.config, name, particles=4)
            self.assertEqual(len(agent.mounts), 4)
            self.assertEqual(set(agent.reference.mounts), set(agent.mounts))

    def test_invalid_fault_sensor_is_rejected(self):
        with self.assertRaises(ValueError):
            LayoutWorld(self.quad, 1001, 'lidar', 'missing')

    def test_duplicate_raw_sensor_is_rejected(self):
        fixture = raw_fixture.TestRawScanHypotheses()
        fixture.setUp()
        packet = fixture.packets()[0]
        with self.assertRaises(ValueError):
            compare_layout_hypotheses(fixture.estimator, [packet, packet], fixture.mounts, fixture.config)

    def test_planner_uses_actual_fov_beam_count_and_range(self):
        from estimation.configurable_geometry import select_layout_motion
        from estimation.active_pipeline import geometry_view
        from simulator.lidar import Lidar
        agent = self.active[0]
        raw = next(row for row in agent.raw_history if row['time_s'] == agent.trigger_time_s)
        configs = {u.sensor_id: u.config for u in read_layout(self.quad)}
        with patch('estimation.configurable_geometry.Lidar', wraps=Lidar) as factory:
            select_layout_motion(geometry_view(agent.reference), raw, agent.mounts, configs)
        self.assertEqual([call.args[0] for call in factory.call_args_list], list(configs.values()))

    def test_missing_sensor_does_not_clear_existing_quarantine(self):
        guard = LayoutScanGuard(ScanGuardConfig(1., .02))
        mounts = {'a': None, 'b': None, 'c': None}
        def alignment(angle):
            return {'offset_deg': angle, 'alignment_rms_m': 0., 'at_search_edge': False}
        with patch('estimation.configurable_geometry.scan_alignment',
                   side_effect=[alignment(5.), alignment(0.), alignment(0.)]):
            guard.observe(None, [{'sensor_id': s} for s in mounts], mounts, 1.)
        self.assertTrue(guard.quarantine_active)
        with patch('estimation.configurable_geometry.scan_alignment',
                   side_effect=[alignment(0.), alignment(0.)]):
            guard.observe(None, [{'sensor_id': s} for s in ('b', 'c')], mounts, 1.5)
        self.assertTrue(guard.quarantine_active)
        self.assertFalse(guard.confirmed)

    def test_guard_unknown_sensor_is_rejected(self):
        guard = LayoutScanGuard(self.config)
        with self.assertRaises(ValueError):
            guard.observe(None, [{'sensor_id': 'unknown'}], {}, 0.)


if __name__ == '__main__':
    unittest.main()
