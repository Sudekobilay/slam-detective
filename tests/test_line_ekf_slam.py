"""Ortak robot-doğru EKF matematiğinin kontrolleri."""

from dataclasses import replace
import unittest
import numpy as np

from estimation.line_ekf_slam import (
    LineEKFSLAM,
    line_initialization,
    line_observation,
)
from perception.line_features import LineFeature
from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.mounting import SensorMount


def example_feature():
    return LineFeature(
        normal=(1.0, 0.0),
        rho_m=2.0,
        start=(2.0, -1.0),
        end=(2.0, 1.0),
        first_beam=0,
        last_beam=10,
        support_count=11,
        length_m=2.0,
        max_point_error_m=0.0,
    )


def numerical_jacobian(function, values):
    epsilon = 1e-6
    columns = []

    for index in range(len(values)):
        offset = np.zeros(len(values))
        offset[index] = epsilon

        columns.append(
            (function(values + offset) - function(values - offset))
            / (2.0 * epsilon)
        )

    return np.column_stack(columns)


class TestLineEKFSLAM(unittest.TestCase):

    def setUp(self):
        self.state = np.array([1.0, 3.0, 0.4, 0.01])
        self.measurement = np.array([2.0, 0.3])
        self.mount = SensorMount(0.25, 0.1, 0.2)

    def test_initialization_jacobians(self):
        landmark, J_robot, J_measurement = line_initialization(
            self.state, self.measurement, self.mount
        )

        numerical_robot = numerical_jacobian(
            lambda state: line_initialization(
                state, self.measurement, self.mount
            )[0],
            self.state,
        )
        numerical_measurement = numerical_jacobian(
            lambda z: line_initialization(
                self.state, z, self.mount
            )[0],
            self.measurement,
        )

        np.testing.assert_allclose(
            J_robot, numerical_robot, atol=1e-8
        )
        np.testing.assert_allclose(
            J_measurement, numerical_measurement, atol=1e-8
        )

        reconstructed = line_observation(
            self.state, landmark, self.mount
        )[0]
        np.testing.assert_allclose(
            reconstructed, self.measurement, atol=1e-9
        )

    def test_observation_jacobians(self):
        landmark = line_initialization(
            self.state, self.measurement, self.mount
        )[0]
        _, H_robot, H_landmark = line_observation(
            self.state, landmark, self.mount
        )

        numerical_robot = numerical_jacobian(
            lambda state: line_observation(
                state, landmark, self.mount
            )[0],
            self.state,
        )
        numerical_landmark = numerical_jacobian(
            lambda line: line_observation(
                self.state, line, self.mount
            )[0],
            landmark,
        )

        np.testing.assert_allclose(
            H_robot, numerical_robot, atol=1e-8
        )
        np.testing.assert_allclose(
            H_landmark, numerical_landmark, atol=1e-8
        )

    def test_initialization_preserves_cross_covariance(self):
        estimator = LineEKFSLAM(
            initial_covariance=np.eye(4) * 0.01
        )
        prior = estimator.P.copy()
        feature = example_feature()

        _, J_robot, _ = line_initialization(
            estimator.state,
            np.array([2.0, 0.0]),
            self.mount,
        )
        estimator.add_line(feature, self.mount)

        self.assertEqual(estimator.P.shape, (6, 6))
        np.testing.assert_allclose(
            estimator.P[:4, 4:6],
            prior @ J_robot.T,
        )
        self.assertGreater(
            np.linalg.norm(estimator.P[:4, 4:6]), 0.0
        )

    def test_first_measurement_not_used_twice(self):
        estimator = LineEKFSLAM()
        feature = example_feature()
        landmark_id = estimator.add_line(feature, self.mount)

        self.assertEqual(estimator.landmark_count, 1)
        self.assertEqual(estimator.line_updates, 0)

        reversed_feature = replace(
            feature,
            normal=(-1.0, 0.0),
            rho_m=-2.0,
        )
        nis = estimator.update_line(
            landmark_id, reversed_feature, self.mount
        )

        self.assertLess(nis, 1e-12)
        self.assertEqual(estimator.line_updates, 1)

    def test_prediction_and_gps_with_augmented_map(self):
        estimator = LineEKFSLAM()
        estimator.add_line(example_feature(), self.mount)

        estimator.predict(
            EncoderMeasurement(0.1, 0.1, 20, 32),
            GyroMeasurement(0.1, 0.1, 0.3, 0.0001),
        )
        estimator.update_gps(
            GPSMeasurement(0.1, True, 0.01, 0.02, 0.0025)
        )

        self.assertEqual(estimator.P.shape, (6, 6))
        self.assertTrue(np.isfinite(estimator.state).all())
        np.testing.assert_allclose(
            estimator.P, estimator.P.T, atol=1e-10
        )
        self.assertGreaterEqual(
            np.linalg.eigvalsh(estimator.P).min(), -1e-8
        )
        self.assertEqual(len(estimator.map_features()), 1)

    def test_invalid_landmark_rejected(self):
        estimator = LineEKFSLAM()

        with self.assertRaises(ValueError):
            estimator.update_line(
                0, example_feature(), self.mount
            )


if __name__ == "__main__":
    unittest.main()