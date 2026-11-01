"""Hipotez karşılaştırmasının geometri ve yan etkisizlik testleri."""

from dataclasses import replace
import math
import unittest
from types import SimpleNamespace

import numpy as np

from estimation.line_ekf_slam import LineEKFSLAM, line_observation
from estimation.mounting_hypotheses import compare_mounting_hypotheses
from simulator.mounting import SensorMount


class TestMountingHypotheses(unittest.TestCase):
    def setUp(self):
        self.slam = LineEKFSLAM.__new__(LineEKFSLAM)
        self.slam.state = np.array([1., 2., .4, .01, 4., .7, 5., 2.])
        self.slam.P = np.eye(8) * .0001
        self.slam.line_R = np.diag([.02**2, math.radians(1.)**2])
        self.slam._anchors = [None, None]
        self.slam.line_updates = 0
        self.mounts = {
            'front': SensorMount(x=.25),
            'rear': SensorMount(x=-.25, yaw=math.pi),
        }

    def observations(self, front_yaw=0., rear_yaw=0., heading=0.):
        observations = []
        robot = self.slam.state[:4].copy()
        robot[2] += math.radians(heading)
        for sensor, nominal in self.mounts.items():
            delta = front_yaw if sensor == 'front' else rear_yaw
            actual = replace(nominal, yaw=nominal.yaw + math.radians(delta))
            for landmark_id in (0, 1):
                index = 4 + 2 * landmark_id
                z, _, _ = line_observation(
                    robot, self.slam.state[index:index+2], actual
                )
                feature = SimpleNamespace(
                    rho_m=float(z[0]),
                    normal=(math.cos(z[1]), math.sin(z[1])),
                )
                observations.append((sensor, landmark_id, feature, nominal))
        return observations

    @staticmethod
    def model(result, name):
        return next(item for item in result['models'] if item['hypothesis'] == name)

    def test_nominal_yaw_models_need_no_offset(self):
        result = compare_mounting_hypotheses(self.slam, self.observations())
        self.assertEqual(result['status'], 'evaluated')
        self.assertAlmostEqual(result['nominal_joint_nis'], 0.)
        for name in ('front_yaw', 'rear_yaw'):
            self.assertEqual(self.model(result, name)['best_offset_deg'], 0.)

    def test_front_yaw_is_recovered_on_known_correspondences(self):
        result = compare_mounting_hypotheses(
            self.slam, self.observations(front_yaw=5.)
        )
        front = self.model(result, 'front_yaw')
        self.assertEqual(front['best_offset_deg'], 5.)
        self.assertAlmostEqual(front['joint_nis'], 0., places=10)
        self.assertGreater(front['nll_improvement_over_nominal'], 0.)
        self.assertLess(front['nll'], self.model(result, 'rear_yaw')['nll'])

    def test_heading_offset_can_explain_shared_error(self):
        result = compare_mounting_hypotheses(
            self.slam, self.observations(heading=3.)
        )
        heading = self.model(result, 'heading_offset')
        self.assertEqual(heading['best_offset_deg'], 3.)
        self.assertAlmostEqual(heading['joint_nis'], 0., places=10)

    def test_missing_sensor_returns_insufficient_support(self):
        front_only = [item for item in self.observations() if item[0] == 'front']
        result = compare_mounting_hypotheses(self.slam, front_only)
        self.assertEqual(result['status'], 'insufficient_support')
        self.assertEqual(result['models'], [])

    def test_comparison_does_not_change_filter(self):
        state, covariance = self.slam.state.copy(), self.slam.P.copy()
        compare_mounting_hypotheses(self.slam, self.observations(front_yaw=-4.))
        np.testing.assert_array_equal(self.slam.state, state)
        np.testing.assert_array_equal(self.slam.P, covariance)
        self.assertEqual(self.slam.line_updates, 0)
        with self.assertRaises(ValueError):
            compare_mounting_hypotheses(self.slam, [], angle_step_deg=0.)


if __name__ == '__main__':
    unittest.main()
