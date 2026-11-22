"""Gösterimde yeniden oynatma, kaynak koruması ve görsel bilgi ayrımı."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from estimation.scan_alignment_guard import ScanGuardConfig
from storage.files import write_json, write_jsonl
from configuration.layouts import examples
from application.layout_service import run_layout
from evaluation.trajectory import evaluate
from storage.files import write_csv
from simulator.mounting import SensorMount
from visualization.active_demo import collect_frames, display_decision, hashes, physical_scan


class TestActiveDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.fixture.name) / 'source'
        cls.folder.mkdir()
        data = examples()['quad']
        config = ScanGuardConfig(1., .02)
        agent, events, commands, records, reason = run_layout(data, config, sensor='left')
        if reason:
            raise RuntimeError(reason)
        write_json(cls.folder/'summary.json', {
            'experiment_id':'exp_40', 'required_horizon_completed':True,
            'algorithm':'ekf', 'trigger_time_s':agent.trigger_time_s,
            'selected_motion':agent.selected_motion,
        })
        write_json(cls.folder/'used_config.json', data)
        write_json(cls.folder/'calibration.json', {'layout':data, 'particles':40, 'config':asdict(config)})
        write_json(cls.folder/'simulation_metadata.json', records['metadata'])
        write_json(cls.folder/'decision.json', agent.decision)
        write_jsonl(cls.folder/'measurements.jsonl', events)
        write_csv(cls.folder/'commands.csv', commands)
        write_csv(cls.folder/'agent_history.csv', agent.history)
        truth = cls.folder/'ground_truth.csv'
        write_csv(truth, records['truth'])
        _, evaluated, _ = evaluate({'reference':agent.reference_path, 'output':agent.path}, truth)
        for name, rows in evaluated.items():
            write_csv(cls.folder/f'{name}_evaluation.csv', rows)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)/'run'
        shutil.copytree(self.folder, self.run)

    def test_replay_matches_saved_and_preserves_inputs(self):
        before = hashes(self.run)
        frames, data, _, summary, _, _, maximum = collect_frames(self.run)
        self.assertEqual(len(data['lidars']), 4)
        self.assertEqual(len(frames), 46)
        self.assertEqual(summary['selected_motion'], 'arc')
        self.assertLess(maximum, 1e-10)
        self.assertEqual(hashes(self.run), before)
        self.assertFalse(frames[0]['geometry_alarm'])
        self.assertTrue(frames[-1]['geometry_alarm'])

    def test_future_decision_is_hidden(self):
        summary = {'trigger_time_s':2.5}
        decision = {'candidates':[{'motion':'arc'}]}
        self.assertIsNone(display_decision(2., summary, decision))
        self.assertEqual(display_decision(2.5, summary, decision), decision)
        self.assertIsNone(display_decision(4., {'trigger_time_s':None}, decision))

    def test_changed_saved_estimate_is_rejected(self):
        path = self.run/'output_evaluation.csv'
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[-1]['x_m'] = float(rows[-1]['x_m']) + .1
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, 'uyuşmuyor'):
            collect_frames(self.run)

    def test_wrong_experiment_is_rejected(self):
        path = self.run/'summary.json'
        summary = json.loads(path.read_text())
        summary['experiment_id'] = 'exp_33'
        write_json(path, summary)
        with self.assertRaisesRegex(ValueError, 'EXP-40'):
            collect_frames(self.run)

    def test_incomplete_source_is_rejected(self):
        path = self.run/'summary.json'
        summary = json.loads(path.read_text())
        summary['required_horizon_completed'] = False
        write_json(path, summary)
        with self.assertRaises(ValueError):
            collect_frames(self.run)

    def test_ground_truth_changes_only_drawing_data(self):
        before = collect_frames(self.run)[0]
        path = self.run/'ground_truth.csv'
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row['x_m'] = float(row['x_m']) + 10.
        write_csv(path, rows)
        after = collect_frames(self.run)[0]
        self.assertEqual([f['reference'] for f in before], [f['reference'] for f in after])
        self.assertEqual([f['output'] for f in before], [f['output'] for f in after])
        self.assertAlmostEqual(after[-1]['truth']['x_m'] - before[-1]['truth']['x_m'], 10.)

    def test_scan_is_held_at_acquisition_time(self):
        frames = collect_frames(self.run)[0]
        frame = next(f for f in frames if abs(f['time_s']-2.2)<1e-9)
        self.assertEqual({s['time_s'] for s in frame['scans'].values()}, {2.})

    def test_physical_fault_transform_is_only_for_target_sensor(self):
        import math
        fault = {'type':'lidar', 'sensor':'left', 'start_s':2., 'yaw_deg':5.}
        scan = {'time_s':2., 'points':[(1.,0.)]}
        truth = {'x_m':0., 'y_m':0., 'theta_rad':0.}
        _, points = physical_scan('left', scan, truth, SensorMount(), fault)
        self.assertAlmostEqual(points[0][0], math.cos(math.radians(5.)))
        self.assertAlmostEqual(points[0][1], math.sin(math.radians(5.)))
        _, points = physical_scan('right', scan, truth, SensorMount(), fault)
        self.assertEqual(points, [(1.,0.)])


if __name__ == '__main__':
    unittest.main()
