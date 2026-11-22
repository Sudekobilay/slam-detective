"""Ölçüme bağlı öneri ve önem ağırlığı testleri."""

from functools import partial
import math
import unittest

import numpy as np

from estimation.fastslam1 import log_gaussian
from estimation.fastslam2 import (
    FastSLAM2Backend,
    conditioned_gaussian,
    state_difference,
)
from configuration.layouts import examples
from experiments.exp_36_sensor_layouts import make_recording, replay_layout
from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.mounting import SensorMount


class TestFastSLAM2(unittest.TestCase):
    def backend(self):
        return FastSLAM2Backend(
            {"front": SensorMount()},
            particle_count=8,
            seed=901,
        )

    def predict(self, backend):
        backend.predict(
            EncoderMeasurement(
                0.1, 0.1, 65, 65
            ),
            GyroMeasurement(
                0.1, 0.1, 0.0, 0.0001
            ),
        )

    def test_measurement_moves_proposal_and_reduces_covariance(self):
        mean = np.zeros(4)
        prior = np.diag([
            0.02, 0.02, 0.01, 0.001
        ])
        h = np.array([
            [1.0, 0.0, 0.0, 0.0]
        ])

        proposed, covariance = conditioned_gaussian(
            mean,
            prior,
            np.array([0.3]),
            h,
            np.array([[0.0025]]),
        )

        self.assertGreater(
            proposed[0],
            0.0,
        )
        self.assertLess(
            proposed[0],
            0.3,
        )
        self.assertLess(
            covariance[0, 0],
            prior[0, 0],
        )
        self.assertGreater(
            np.linalg.eigvalsh(
                covariance
            ).min(),
            0.0,
        )

    def test_importance_identity_for_linear_gaussian_model(self):
        mean = np.zeros(4)
        prior = np.diag([
            0.02, 0.02, 0.01, 0.001
        ])
        h = np.array([
            [1.0, 0.0, 0.0, 0.0]
        ])
        noise = np.array([[0.0025]])
        z = np.array([0.3])

        proposed, covariance = conditioned_gaussian(
            mean,
            prior,
            z,
            h,
            noise,
        )

        sample = proposed + np.array([
            0.01, -0.02, 0.01, 0.001
        ])

        corrected = (
            log_gaussian(
                z - h @ sample,
                noise,
            )
            + log_gaussian(
                sample - mean,
                prior,
            )
            - log_gaussian(
                sample - proposed,
                covariance,
            )
        )

        evidence = log_gaussian(
            z - h @ mean,
            h @ prior @ h.T + noise,
        )

        self.assertAlmostEqual(
            corrected,
            evidence,
            places=10,
        )

    def test_angle_difference_wraps(self):
        first = np.array([
            0.0, 0.0,
            math.pi - 0.01,
            0.0,
        ])
        second = np.array([
            0.0, 0.0,
            -math.pi + 0.01,
            0.0,
        ])

        self.assertAlmostEqual(
            state_difference(
                first,
                second,
            )[2],
            -0.02,
        )

    def test_missing_gps_does_not_consume_pending_proposal(self):
        backend = self.backend()
        self.predict(backend)

        backend.update_gps(
            GPSMeasurement(
                0.1,
                False,
                None,
                None,
                0.0025,
            )
        )

        self.assertIsNotNone(
            backend.pending
        )
        self.assertEqual(
            len(backend.proposal_history),
            0,
        )

    def test_gps_conditions_particles_and_logs_p_over_q(self):
        backend = self.backend()
        self.predict(backend)

        backend.update_gps(
            GPSMeasurement(
                0.1,
                True,
                1.05,
                3.0,
                0.0025,
            )
        )

        row = backend.proposal_history[-1]

        self.assertEqual(
            row["conditioned_particles"],
            8,
        )
        self.assertLess(
            row["mean_proposal_trace"],
            row["mean_prior_trace"],
        )
        self.assertTrue(
            math.isfinite(
                row["mean_log_p_over_q"]
            )
        )
        self.assertIsNone(backend.pending)

    def test_second_sensor_does_not_reuse_same_motion_prior(self):
        backend = self.backend()
        self.predict(backend)

        backend.update_gps(
            GPSMeasurement(
                0.1,
                True,
                1.0,
                3.0,
                0.0025,
            )
        )

        backend.update_scan(
            "front",
            [
                (2.0, -1.0 + index * 0.1)
                for index in range(21)
            ],
        )

        self.assertEqual(
            len(backend.proposal_history),
            1,
        )

    def test_existing_line_can_condition_proposal(self):
        backend = self.backend()

        for particle in backend.particles:
            particle.state[:] = [
                1.0, 3.0, 0.0, 0.0
            ]

        points = [
            (2.0, -1.0 + index * 0.1)
            for index in range(21)
        ]

        backend.update_scan(
            "front",
            points,
        )

        self.predict(backend)

        backend.update_scan(
            "front",
            points,
        )

        row = backend.proposal_history[-1]

        self.assertEqual(
            row["conditioned_particles"],
            8,
        )
        self.assertEqual(
            row["source"],
            "lidar/front",
        )

    def test_new_map_without_match_uses_motion_proposal(self):
        backend = self.backend()
        self.predict(backend)

        backend.update_scan(
            "front",
            [
                (2.0, -1.0 + index * 0.1)
                for index in range(21)
            ],
        )

        self.assertEqual(
            backend.proposal_history[-1][
                "conditioned_particles"
            ],
            0,
        )
        self.assertGreater(
            backend.landmark_count,
            0,
        )

    def test_duplicate_gps_does_not_resample_again(self):
        backend = self.backend()
        self.predict(backend)

        packet = GPSMeasurement(
            0.1,
            True,
            1.0,
            3.0,
            0.0025,
        )

        backend.update_gps(packet)
        count = len(
            backend.proposal_history
        )

        with self.assertRaises(ValueError):
            backend.update_gps(packet)

        self.assertEqual(
            len(backend.proposal_history),
            count,
        )

    def test_complete_replay_is_reproducible(self):
        data = {
            **examples()["dual"],
            "gps_enabled": False,
        }

        events, _ = make_recording(data)

        factory = partial(
            FastSLAM2Backend,
            particle_count=8,
            seed=902,
        )

        first, path1 = replay_layout(
            events,
            data,
            backend_factory=factory,
        )
        second, path2 = replay_layout(
            events,
            data,
            backend_factory=factory,
        )

        self.assertEqual(path1, path2)
        self.assertEqual(len(path1), 41)

        self.assertTrue(
            np.isfinite(
                first.weights
            ).all()
        )
        self.assertAlmostEqual(
            first.weights.sum(),
            1.0,
        )

        self.assertTrue(any(
            row["conditioned_particles"] > 0
            for row in first.proposal_history
        ))

        self.assertEqual(
            first.landmark_count,
            second.landmark_count,
        )


if __name__ == "__main__":
    unittest.main()