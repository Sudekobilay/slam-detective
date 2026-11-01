import unittest

import numpy as np

from estimation.gps_gate import GatedLineEKFSLAM, GatedPoseEKF
from simulator.gps import GPSMeasurement


class TestGPSGate(unittest.TestCase):
    def test_consistent_measurement_accepted(self):
        for estimator_type in (GatedPoseEKF, GatedLineEKFSLAM):
            with self.subTest(estimator=estimator_type.__name__):
                estimator = estimator_type()
                estimator.update_gps(
                    GPSMeasurement(0.0, True, 0.0, 0.0, 0.0025)
                )

                self.assertEqual(estimator.gps_updates, 1)
                self.assertEqual(estimator.gps_rejected, 0)
                self.assertTrue(
                    estimator.gps_gate_log[-1]["accepted"]
                )

    def test_outlier_rejected_without_changing_estimate(self):
        for estimator_type in (GatedPoseEKF, GatedLineEKFSLAM):
            with self.subTest(estimator=estimator_type.__name__):
                estimator = estimator_type()
                state_before = estimator.state.copy()
                covariance_before = estimator.P.copy()

                measurement = GPSMeasurement(
                    0.0, True, 1.0, -1.0, 0.0025
                )
                estimator.update_gps(measurement)

                np.testing.assert_array_equal(
                    estimator.state, state_before
                )
                np.testing.assert_array_equal(
                    estimator.P, covariance_before
                )
                self.assertEqual(estimator.gps_updates, 0)
                self.assertEqual(estimator.gps_rejected, 1)

                # Reddedilen paket de ikinci kez işlenmemeli.
                with self.assertRaises(ValueError):
                    estimator.update_gps(measurement)

    def test_missing_measurement_is_not_outlier(self):
        for estimator_type in (GatedPoseEKF, GatedLineEKFSLAM):
            with self.subTest(estimator=estimator_type.__name__):
                estimator = estimator_type()
                estimator.update_gps(
                    GPSMeasurement(0.0, False, None, None, 0.0025)
                )

                self.assertEqual(estimator.gps_missing, 1)
                self.assertEqual(estimator.gps_rejected, 0)
                self.assertEqual(estimator.gps_gate_log, [])


if __name__ == "__main__":
    unittest.main()