"""Kayıt biçimi, gerçek odometri dönüşümü ve ortak çekirdeğe bağlanma."""
import gzip
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset_loaders.carmen import CarmenLoader, relative_pose, laser_mount
from estimation.active_pipeline import make_backend
from estimation.recorded_odometry import odometry_prediction, predict_recorded_odometry
from experiments.exp_41_real_data import replay_recording
from simulator.mounting import SensorMount
from simulator.robot import Pose


def fixture_line(timestamp, x=0., angle=0., ranges=None):
    ranges = ranges or [2.] * 360
    # Sabit, öne 0.1 m takılmış sensör.
    laser_x, laser_y = x + .1*math.cos(angle), .1*math.sin(angle)
    return 'FLASER 360 ' + ' '.join(map(str, ranges)) + f' {laser_x} {laser_y} {angle} {x} 0 {angle} {timestamp} host 0\n'


class TestRealData(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.clf'
        self.path.write_text('# fixture, not real dataset\n' + fixture_line(10.) + fixture_line(10.5, .1), encoding='utf-8')

    def test_pose_relation_and_mount(self):
        scan = next(iter(CarmenLoader(self.path)))
        self.assertAlmostEqual(laser_mount(scan).x, .1)
        delta = relative_pose(Pose(1., 2., math.pi/2), Pose(1., 3., math.pi))
        self.assertAlmostEqual(delta.x, 1.)
        self.assertAlmostEqual(delta.y, 0.)
        self.assertAlmostEqual(delta.theta, math.pi/2)

    def test_profile_has_half_degree_increment(self):
        scan = next(iter(CarmenLoader(self.path)))
        self.assertEqual(len(scan.points), 360)
        self.assertAlmostEqual(scan.points[0][0], 0.)
        self.assertAlmostEqual(scan.points[0][1], -2.)
        self.assertAlmostEqual(math.atan2(*reversed(scan.points[-1])), math.radians(89.5))

    def test_invalid_ranges_are_missing_points(self):
        values = [2.] * 360
        values[:4] = [0., -1., float('nan'), 80.]
        self.path.write_text(fixture_line(10., ranges=values))
        self.assertEqual(next(iter(CarmenLoader(self.path))).points[:4], (None,)*4)

    def test_gzip_matches_plain(self):
        compressed = self.path.with_suffix('.gz')
        with gzip.open(compressed, 'wt') as handle:
            handle.write(self.path.read_text())
        self.assertEqual(list(CarmenLoader(self.path)), list(CarmenLoader(compressed)))

    def test_bad_count_and_repeated_time_rejected(self):
        self.path.write_text(fixture_line(10.) + fixture_line(10.))
        with self.assertRaises(ValueError):
            list(CarmenLoader(self.path))
        with self.assertRaises(ValueError):
            list(CarmenLoader(self.path, expected_beams=180))

    def test_direct_increment_and_jacobian(self):
        state = np.array([1., 3., .3, .01])
        delta = Pose(.2, -.1, .05)
        predicted, jacobian, _ = odometry_prediction(state, delta)
        epsilon = 1e-6
        for index in range(4):
            plus, minus = state.copy(), state.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric = (odometry_prediction(plus, delta)[0] - odometry_prediction(minus, delta)[0]) / (2*epsilon)
            np.testing.assert_allclose(jacobian[:, index], numeric, atol=1e-8)
        self.assertAlmostEqual(predicted[3], .01)

    def test_no_fake_gyro_bias_correction_on_odometry(self):
        backend = make_backend('ekf', mounts={'recorded_lidar': SensorMount()})
        backend.estimator.state[3] = 2.
        predict_recorded_odometry(backend, Pose(.1, 0., .2), .5)
        self.assertAlmostEqual(backend.estimator.pose.theta, .2)
        self.assertAlmostEqual(backend.estimator.state[3], 2.)
        self.assertGreater(np.linalg.eigvalsh(backend.estimator.P).min(), 0.)

    def test_three_backends_replay_and_source_is_unchanged(self):
        before = self.path.read_bytes()
        for algorithm in ('ekf', 'fastslam1', 'fastslam2'):
            result = replay_recording(self.path, algorithm=algorithm, max_scans=2, stride=1, particles=4)
            self.assertEqual(len(result[1]), 2)
            self.assertEqual(result[-1]['duration_s'], .5)
            self.assertTrue(all(math.isfinite(r['x_m']) for r in result[1]))
        self.assertEqual(before, self.path.read_bytes())

    def test_fastslam2_retains_measurement_conditioned_proposal(self):
        backend = make_backend('fastslam2', particles=4, mounts={'recorded_lidar': SensorMount()})
        predict_recorded_odometry(backend, Pose(.1, .02, .1), .5)
        self.assertEqual(len(backend.pending), 4)
        self.assertTrue(all(np.linalg.eigvalsh(prior).min() > 0 for _, prior in backend.pending))

    def test_joint_robot_map_covariance_matches_full_jacobian(self):
        backend = make_backend('ekf', mounts={})
        backend.estimator.state = np.array([1., 3., .2, 0., 2., .4])
        rng = np.random.default_rng(1)
        root = rng.normal(size=(6, 6))
        covariance = root @ root.T + np.eye(6)
        backend.estimator.P = covariance.copy()
        delta = Pose(.2, -.1, .05)
        _, jacobian, rotation = odometry_prediction(backend.estimator.state[:4], delta)
        full = np.eye(6)
        full[:4, :4] = jacobian
        expected = full @ covariance @ full.T
        expected[:4, :4] += rotation @ np.diag([.02**2*.5]*2 + [.03**2*.5, .001**2*.5]) @ rotation.T
        predict_recorded_odometry(backend, delta, .5)
        np.testing.assert_allclose(backend.estimator.P, expected, atol=1e-12)

    def test_nonpositive_time_rejected(self):
        backend = make_backend('ekf', mounts={})
        with self.assertRaises(ValueError):
            predict_recorded_odometry(backend, Pose(), 0.)


if __name__ == '__main__':
    unittest.main()
