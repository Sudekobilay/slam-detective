"""EKF matematiğinin bağımsız kontrolleri."""

import unittest
import numpy as np

from estimation.pose_ekf import PoseEKF, motion_prediction
from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement


class TestPoseEKF(unittest.TestCase):

    def test_state_jacobian(self):
        state = np.array([0.2, 0.3, 0.4, 0.01])
        _, analytical, _ = motion_prediction(
            state, 0.08, 0.03, 0.1
        )
        numerical = np.zeros((4, 4))
        epsilon = 1e-6

        for index in range(4):
            offset = np.zeros(4)
            offset[index] = epsilon
            plus = motion_prediction(
                state + offset, 0.08, 0.03, 0.1
            )[0]
            minus = motion_prediction(
                state - offset, 0.08, 0.03, 0.1
            )[0]
            numerical[:, index] = (
                plus - minus
            ) / (2 * epsilon)

        np.testing.assert_allclose(
            analytical, numerical, atol=1e-8, rtol=1e-6
        )

    def test_control_jacobian(self):
        state = np.array([0.2, 0.3, 0.4, 0.01])

        # Düz ve dönüşlü hareket yakınında ayrı kontrol.
        for measured_turn in (0.001, 0.03):
            with self.subTest(turn=measured_turn):
                controls = np.array([0.08, measured_turn])
                _, _, analytical = motion_prediction(
                    state, *controls, 0.1
                )
                numerical = np.zeros((4, 2))
                epsilon = 1e-6

                for index in range(2):
                    offset = np.zeros(2)
                    offset[index] = epsilon
                    plus = motion_prediction(
                        state, *(controls + offset), 0.1
                    )[0]
                    minus = motion_prediction(
                        state, *(controls - offset), 0.1
                    )[0]
                    numerical[:, index] = (
                        plus - minus
                    ) / (2 * epsilon)

                np.testing.assert_allclose(
                    analytical, numerical,
                    atol=1e-8, rtol=1e-6,
                )

    def test_gps_update_matches_scalar_kalman(self):
        estimator = PoseEKF(
            initial_covariance=np.diag([1.0, 1.0, 0.01, 0.01])
        )

        estimator.update_gps(
            GPSMeasurement(0.0, True, 2.0, 4.0, 1.0)
        )

        # P=1, R=1 -> K=0.5.
        self.assertAlmostEqual(estimator.state[0], 1.0)
        self.assertAlmostEqual(estimator.state[1], 2.0)
        self.assertAlmostEqual(estimator.P[0, 0], 0.5)
        self.assertAlmostEqual(estimator.P[1, 1], 0.5)
        self.assertAlmostEqual(
            estimator.innovations[-1]["nis"], 10.0
        )

    def test_missing_gps_does_not_change_state(self):
        estimator = PoseEKF()
        state = estimator.state.copy()
        covariance = estimator.P.copy()

        result = estimator.update_gps(
            GPSMeasurement(0.0, False, None, None, 0.01)
        )

        self.assertFalse(result)
        np.testing.assert_array_equal(estimator.state, state)
        np.testing.assert_array_equal(estimator.P, covariance)

    def test_wrong_time_rejected(self):
        estimator = PoseEKF()
        previous = estimator.state.copy()

        with self.assertRaises(ValueError):
            estimator.update_gps(
                GPSMeasurement(1.0, True, 1.0, 1.0, 0.01)
            )

        np.testing.assert_array_equal(estimator.state, previous)

    def test_prediction_covariance_is_symmetric_psd(self):
        estimator = PoseEKF()

        for index in range(1, 21):
            time_s = index * 0.1

            estimator.predict(
                EncoderMeasurement(time_s, 0.1, 20, 32),
                GyroMeasurement(time_s, 0.1, 0.3, 0.0001),
            )

            self.assertTrue(np.isfinite(estimator.state).all())
            np.testing.assert_allclose(
                estimator.P, estimator.P.T, atol=1e-12
            )
            self.assertGreaterEqual(
                np.linalg.eigvalsh(estimator.P).min(), -1e-10
            )


if __name__ == "__main__":
    unittest.main()