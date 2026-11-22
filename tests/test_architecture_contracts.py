import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from application.cli import main
from configuration.layouts import examples, read_layout
from domain.contracts import SlamBackend, PipelineMode
from domain.geometry import Pose, RobotGeometry
from estimation.backends import make_backend
from estimation.active_pipeline import ActivePipeline
from estimation.scan_alignment_guard import ScanGuardConfig
from storage.files import write_json, write_jsonl, write_csv, read_json
from storage.provenance import source_hashes
from storage.recorders import JsonlRecorder
from storage.runs import RunSession, validate_run


ROOT = Path(__file__).resolve().parents[1]


class TestBackendContract(unittest.TestCase):
    def test_all_backends_support_same_contract_and_custom_prior(self):
        mounts = {u.sensor_id: u.actual_mount for u in read_layout(examples()['quad'])}
        pose, geometry = Pose(2., 2.5, -.4), RobotGeometry(.06, .06, .35)
        for name in ('ekf', 'fastslam1', 'fastslam2'):
            with self.subTest(algorithm=name):
                backend = make_backend(name, 4, mounts=mounts, initial_pose=pose,
                                       initial_covariance=np.zeros((4, 4)), geometry=geometry)
                self.assertIsInstance(backend, SlamBackend)
                self.assertEqual(backend.geometry_snapshot().pose, pose)
                self.assertEqual(backend.geometry_snapshot().geometry, geometry)
                distribution = backend.reference_distribution()
                np.testing.assert_allclose(distribution.state[:3], [pose.x, pose.y, pose.theta])
                self.assertEqual(distribution.time_s, backend.clock_s)
                self.assertTrue(np.isfinite(distribution.P).all())
                self.assertFalse(backend.capabilities.diagnostic_performance_certified)

    def test_both_reference_and_fusion_receive_initial_prior(self):
        for name in ('ekf', 'fastslam1', 'fastslam2'):
            agent = ActivePipeline(name, ScanGuardConfig(1., .02), particles=4,
                                   initial_pose=Pose(2., 2., .3),
                                   initial_covariance=np.zeros((4, 4)), use_gps=True)
            np.testing.assert_allclose(agent.reference.reference_distribution().state,
                                       agent.fusion.reference_distribution().state)
            self.assertAlmostEqual(agent.reference.reference_distribution().state[0], 2.)

    def test_bad_initial_covariance_rejected_by_all_backends(self):
        for name in ('ekf', 'fastslam1', 'fastslam2'):
            with self.subTest(algorithm=name), self.assertRaises(ValueError):
                make_backend(name, 4, initial_covariance=-np.eye(4))

    def test_mode_transition_cannot_restart_ended_diagnosis(self):
        agent = ActivePipeline('ekf', ScanGuardConfig(1., .02))
        agent.mode = PipelineMode.DIAGNOSING
        agent.mode = PipelineMode.DIAGNOSIS_WINDOW_ENDED
        with self.assertRaises(RuntimeError):
            agent.mode = PipelineMode.NORMAL
        with self.assertRaises(ValueError):
            agent.mode = 'TYPO'


class TestStorageContract(unittest.TestCase):
    def test_failed_json_write_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.json'
            write_json(path, {'old': 1})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_json(path, {'invalid': float('nan')})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_utf8_json_csv_and_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            row = {'label': 'Ölçüm — dönüş', 'available': True}
            write_json(path / 'a.json', row)
            self.assertEqual(read_json(path / 'a.json'), row)
            write_jsonl(path / 'b.jsonl', [row])
            self.assertEqual(json.loads((path / 'b.jsonl').read_text(encoding='utf-8')), row)
            write_csv(path / 'c.csv', [row])
            self.assertIn('Ölçüm', (path / 'c.csv').read_text(encoding='utf-8'))

    def test_disk_recorder_bounds_recent_rows_but_retains_disk_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = JsonlRecorder(Path(tmp) / 'history', recent_rows=2)
            stream = recorder.stream('path')
            for i in range(5):
                stream.append({'time_s': i})
            self.assertEqual(len(stream), 2)
            self.assertEqual(len((Path(tmp) / 'history/path.jsonl').read_text().splitlines()), 5)
            with self.assertRaises(FileExistsError):
                JsonlRecorder(Path(tmp) / 'history')

    def test_provenance_covers_config_and_evaluation_but_not_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('config/a.json', 'evaluation/a.py', 'results/a.py'):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text('{}')
            hashes = source_hashes(root)
            self.assertIn('config/a.json', hashes)
            self.assertIn('evaluation/a.py', hashes)
            self.assertNotIn('results/a.py', hashes)
            (root / 'config/a.json').write_text('{"changed":true}')
            self.assertNotEqual(hashes, source_hashes(root))

    def test_completed_failed_and_interrupted_run_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'complete'
            with RunSession(folder, root, {'seed': 1}, experiment_id='test') as session:
                session.checkpoint(4)
                session.finish(complete=True)
            self.assertEqual(read_json(folder / 'run_manifest.json')['status'], 'COMPLETED')
            self.assertEqual(read_json(folder / 'run_manifest.json')['completed_cases'], 4)
            folder = root / 'failed'
            with self.assertRaises(ArithmeticError):
                with RunSession(folder, root, {}, experiment_id='test'):
                    raise ArithmeticError('injected')
            self.assertEqual(read_json(folder / 'run_manifest.json')['status'], 'FAILED')
            folder = root / 'interrupted'
            with self.assertRaises(KeyboardInterrupt):
                with RunSession(folder, root, {}, experiment_id='test'):
                    raise KeyboardInterrupt()
            self.assertEqual(read_json(folder / 'run_manifest.json')['status'], 'PARTIAL')

    def test_incomplete_run_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'run'
            with RunSession(folder, root, {}, experiment_id='exp_43') as session:
                write_json(folder / 'summary.json', {'experiment_id': 'exp_43', 'all_horizons_completed': False})
                session.finish(complete=False)
            with self.assertRaises(ValueError):
                validate_run(folder, experiment_id='exp_43')
            self.assertFalse(validate_run(folder, experiment_id='exp_43', allow_partial=True)['all_horizons_completed'])

    def test_explicit_partial_exit_keeps_partial_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                with RunSession(root / 'run', root, {}, experiment_id='test') as session:
                    session.finish(complete=False)
                    raise RuntimeError('incomplete')
            self.assertEqual(read_json(root / 'run/run_manifest.json')['status'], 'PARTIAL')


class TestArchitectureBoundaries(unittest.TestCase):
    def test_shared_layers_do_not_import_numbered_experiments(self):
        for area in ('domain', 'application', 'configuration', 'evaluation', 'storage',
                     'estimation', 'simulator', 'perception', 'analytics', 'visualization'):
            for path in (ROOT / area).glob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or '').startswith('experiments.'), str(path))
                    elif isinstance(node, ast.Import):
                        self.assertFalse(any(n.name.startswith('experiments.') for n in node.names), str(path))

    def test_domain_has_no_simulation_or_runner_imports(self):
        for path in (ROOT / 'domain').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('simulator.', 'experiments.', 'application.')))

    def test_cli_does_not_mutate_global_arguments(self):
        original = list(sys.argv)
        received = []
        fake = SimpleNamespace(main=lambda argv: received.append(argv))
        with patch('application.cli.importlib.import_module', return_value=fake):
            main(['robustness', '--smoke'])
        self.assertEqual(received, [['--smoke']])
        self.assertEqual(sys.argv, original)

    def test_gate_compatibility_is_same_class(self):
        from estimation.lidar_update_hold import LidarUpdateHold
        from experiments.lidar_update_hold import LidarUpdateHold as LegacyHold
        self.assertIs(LidarUpdateHold, LegacyHold)
