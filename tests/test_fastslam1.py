"""FastSLAM parçacık ve harita testleri."""

import copy
from functools import partial
import math
import unittest

import numpy as np

from estimation.fastslam1 import (
    FastSLAM1Backend,
    log_gaussian,
)
from experiments.exp_36_sensor_layouts import (
    examples,
    make_recording,
    replay_layout,
)
from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.mounting import SensorMount


class TestFastSLAM1(unittest.TestCase):
    def backend(self, seed=901, count=8):
        return FastSLAM1Backend(
            {"front": SensorMount()},
            particle_count=count,
            seed=seed,
        )

    def wall_points(self):
        return [
            (2.0, -1.0 + index * 0.1)
            for index in range(21)
        ]

    def test_invalid_particle_count_rejected(self):
        for count in (0, 1, 2.5, True):
            with self.assertRaises(ValueError):
                self.backend(count=count)

    def test_initial_samples_reproducible_and_nonidentical(self):
        first = self.backend()
        second = self.backend()

        self.assertTrue(np.array_equal(
            np.asarray([
                particle.state
                for particle in first.particles
            ]),
            np.asarray([
                particle.state
                for particle in second.particles
            ]),
        ))

        self.assertGreater(
            np.var([
                particle.state[2]
                for particle in first.particles
            ]),
            0.0,
        )

    def test_prediction_advances_and_rejects_duplicate_time(self):
        backend = self.backend()

        encoder = EncoderMeasurement(
            0.1, 0.1, 65, 65
        )
        gyro = GyroMeasurement(
            0.1, 0.1, 0.0, 0.0001
        )

        backend.predict(encoder, gyro)

        self.assertEqual(
            backend.time_s,
            0.1,
        )
        self.assertGreater(
            backend.pose.x,
            1.0,
        )

        with self.assertRaises(ValueError):
            backend.predict(encoder, gyro)

    def test_gaussian_normalizer(self):
        self.assertAlmostEqual(
            log_gaussian(
                np.zeros(2),
                np.eye(2),
            ),
            -math.log(2.0 * math.pi),
        )

    def test_gps_likelihood_favors_closer_particle(self):
        backend = self.backend(count=2)

        backend.particles[0].state[:2] = [
            1.0, 3.0
        ]
        backend.particles[1].state[:2] = [
            1.05, 3.0
        ]

        backend.update_gps(
            GPSMeasurement(
                0.0,
                True,
                1.0,
                3.0,
                0.0025,
            )
        )

        self.assertGreater(
            backend.weights[0],
            backend.weights[1],
        )

    def test_missing_gps_does_not_change_weights(self):
        backend = self.backend()
        before = backend.weights.copy()

        result = backend.update_gps(
            GPSMeasurement(
                0.0,
                False,
                None,
                None,
                0.0025,
            )
        )

        self.assertFalse(result)
        self.assertTrue(np.array_equal(
            before,
            backend.weights,
        ))
        self.assertEqual(
            backend.gps_missing,
            1,
        )

    def test_each_particle_has_independent_two_dimensional_landmark_filter(self):
        backend = self.backend()

        backend.update_scan(
            "front",
            self.wall_points(),
        )

        self.assertTrue(all(
            len(particle.landmarks) == 1
            for particle in backend.particles
        ))

        landmark = (
            backend.particles[0].landmarks[0]
        )

        self.assertEqual(
            landmark.mean.shape,
            (2,),
        )
        self.assertEqual(
            landmark.covariance.shape,
            (2, 2),
        )

        before = (
            backend.particles[1]
            .landmarks[0].mean.copy()
        )

        landmark.mean[0] += 0.1

        self.assertTrue(np.array_equal(
            backend.particles[1].landmarks[0].mean,
            before,
        ))

    def test_repeated_measurement_reduces_conditional_map_covariance(self):
        backend = self.backend()

        for particle in backend.particles:
            particle.state[:] = [
                1.0, 3.0, 0.0, 0.0
            ]

        backend.update_scan(
            "front",
            self.wall_points(),
        )

        initial = np.trace(
            backend.particles[0]
            .landmarks[0].covariance
        )

        backend.update_scan(
            "front",
            self.wall_points(),
        )

        self.assertLess(
            np.trace(
                backend.particles[0]
                .landmarks[0].covariance
            ),
            initial,
        )

    def test_resampling_deep_copies_maps(self):
        backend = self.backend()

        backend.update_scan(
            "front",
            self.wall_points(),
        )

        backend.log_weights[:] = -1000.0
        backend.log_weights[0] = 0.0

        backend._resample_if_needed("test")

        self.assertEqual(
            backend.resample_count,
            1,
        )

        before = (
            backend.particles[1]
            .landmarks[0].mean.copy()
        )

        backend.particles[0].landmarks[0].mean[0] += 1.0

        self.assertTrue(np.array_equal(
            backend.particles[1].landmarks[0].mean,
            before,
        ))
        self.assertAlmostEqual(
            backend.weights.sum(),
            1.0,
        )

    def test_heading_uses_circular_mean(self):
        backend = self.backend(count=2)

        backend.particles[0].state[2] = (
            math.pi - 0.01
        )
        backend.particles[1].state[2] = (
            -math.pi + 0.01
        )

        self.assertAlmostEqual(
            abs(backend.pose.theta),
            math.pi,
        )

    def test_unknown_sensor_rejected(self):
        with self.assertRaises(ValueError):
            self.backend().update_scan(
                "unknown",
                self.wall_points(),
            )

    def test_full_replay_is_reproducible_and_preserves_input(self):
        data = {
            **examples()["dual"],
            "gps_enabled": False,
        }

        events, _ = make_recording(data)
        original = copy.deepcopy(events)

        factory = partial(
            FastSLAM1Backend,
            particle_count=8,
            seed=903,
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
        self.assertEqual(events, original)
        self.assertEqual(len(path1), 41)

        self.assertGreater(
            first.landmark_count,
            0,
        )

        self.assertTrue(all(
            1.0 - 1e-9
            <= row["ess_before_resampling"]
            <= 8.0 + 1e-9
            for row in first.history
        ))

        self.assertEqual(
            first.landmark_count,
            second.landmark_count,
        )


if __name__ == "__main__":
    unittest.main()