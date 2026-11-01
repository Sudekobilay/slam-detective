"""LiDAR inovasyon sorgusunun matematiği ve yan etkisizliği."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from estimation.line_ekf_slam import LineEKFSLAM, line_observation
from simulator.mounting import SensorMount
from simulator.robot import wrap_angle


class TestLineInnovation(unittest.TestCase):
    def setUp(self):
        # Yalnız ölçüm hesabını sınayan, elle kurulmuş ortak durum.
        self.slam = LineEKFSLAM.__new__(LineEKFSLAM)
        self.slam.state = np.array([1.0, 2.0, 0.4, 0.01, 4.0, 0.7])
        generator = np.random.default_rng(13)
        matrix = generator.normal(size=(6, 6))
        self.slam.P = matrix @ matrix.T * 0.001 + np.eye(6) * 0.0001
        self.slam.line_R = np.diag([0.02**2, math.radians(1.0)**2])
        self.slam._anchors = [None]
        self.slam.line_updates = 0
        self.mount = SensorMount(x=0.25, y=0.1, yaw=0.2)

    @staticmethod
    def feature(rho, alpha):
        return SimpleNamespace(
            rho_m=rho,
            normal=(math.cos(alpha), math.sin(alpha)),
        )

    def nominal_measurement(self):
        expected, _, _ = line_observation(
            self.slam.state[:4], self.slam.state[4:6], self.mount
        )
        return expected

    def test_query_does_not_change_filter(self):
        expected = self.nominal_measurement()
        feature = self.feature(expected[0] + 0.03, expected[1] + 0.02)
        state = self.slam.state.copy()
        covariance = self.slam.P.copy()
        first = self.slam.line_innovation(0, feature, self.mount)
        second = self.slam.line_innovation(0, feature, self.mount)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(self.slam.state, state)
        np.testing.assert_array_equal(self.slam.P, covariance)
        self.assertEqual(self.slam.line_updates, 0)
        with self.assertRaises(ValueError):
            self.slam.line_innovation(1, feature, self.mount)

    def test_covariance_includes_robot_map_cross_terms(self):
        expected, robot_H, line_H = line_observation(
            self.slam.state[:4], self.slam.state[4:6], self.mount
        )
        feature = self.feature(expected[0] + 0.03, expected[1] + 0.02)
        result = self.slam.line_innovation(0, feature, self.mount)
        H = np.hstack([robot_H, line_H])
        S = H @ self.slam.P @ H.T + self.slam.line_R
        residual = np.array([0.03, 0.02])
        np.testing.assert_allclose(result['innovation_covariance'], S)
        self.assertAlmostEqual(
            result['nis'], float(residual @ np.linalg.solve(S, residual))
        )

    def test_opposite_normal_gives_same_innovation(self):
        expected = self.nominal_measurement()
        rho, alpha = expected + np.array([0.03, 0.02])
        first = self.slam.line_innovation(
            0, self.feature(rho, alpha), self.mount
        )
        second = self.slam.line_innovation(
            0, self.feature(-rho, alpha + math.pi), self.mount
        )
        for field in ('residual_rho_m', 'residual_alpha_rad', 'nis'):
            self.assertAlmostEqual(first[field], second[field], places=10)

    def test_angular_residual_wraps_at_pi(self):
        self.slam.state[2] = 0.0
        self.slam.state[5] = math.pi - 0.01
        mount = SensorMount()
        expected, _, _ = line_observation(
            self.slam.state[:4], self.slam.state[4:6], mount
        )
        feature = self.feature(expected[0], -math.pi + 0.01)
        result = self.slam.line_innovation(0, feature, mount)
        self.assertAlmostEqual(result['residual_alpha_rad'], 0.02)

    def test_query_nis_equals_subsequent_update_nis(self):
        expected = self.nominal_measurement()
        feature = self.feature(expected[0] + 0.03, expected[1] + 0.02)
        result = self.slam.line_innovation(0, feature, self.mount)
        nis = self.slam.update_line(0, feature, self.mount)
        self.assertAlmostEqual(result['nis'], nis)
        self.assertEqual(self.slam.line_updates, 1)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.slam.P).min(), -1e-10)


if __name__ == '__main__':
    unittest.main()
