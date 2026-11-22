"""Teşhis penceresinde LiDAR'ın kestirimi değiştirmediğini doğrular."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp_24_motion_comparison import generate_scenario
from storage.files import write_jsonl
from experiments.exp_25_frozen_reference import run_frozen_window


class TestFrozenReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.results = {}
        cls.paths = {}
        cls.hashes = {}
        for name, yaw in (('healthy', 0.), ('faulty', 2.)):
            folder = Path(cls.temp.name) / name
            folder.mkdir()
            source = folder / 'measurements.jsonl'
            write_jsonl(source, generate_scenario('arc', yaw)['events'])
            cls.paths[name] = source
            cls.hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            cls.results[name] = run_frozen_window(source, folder)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_map_and_update_count_are_frozen(self):
        for result in self.results.values():
            self.assertTrue(all(result['checks'].values()), result['checks'])
            self.assertEqual(len(result['path']), 41)
            self.assertAlmostEqual(result['path'][-1]['time_s'], 4.)
            self.assertEqual([r['time_s'] for r in result['hypotheses']], [2., 2.5, 3., 3.5, 4.])

    def test_fault_changes_measurements_but_not_frozen_window_pose(self):
        self.assertNotEqual(self.hashes['healthy'], self.hashes['faulty'])
        self.assertEqual(self.results['healthy']['path'], self.results['faulty']['path'])
        np.testing.assert_array_equal(
            self.results['healthy']['slam'].state,
            self.results['faulty']['slam'].state,
        )
        self.assertNotEqual(
            self.results['healthy']['innovation_rows'],
            self.results['faulty']['innovation_rows'],
        )

    def test_input_recordings_remain_unchanged(self):
        for name, path in self.paths.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.hashes[name])


if __name__ == '__main__':
    unittest.main()
