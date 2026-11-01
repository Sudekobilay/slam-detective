"""Ölçüme bağlı öneriyle çizgi FastSLAM 2.0 uyarlaması.

Son hareket adımı için yerel Gaussian öncel kullanılır.
İlk kullanılabilir GPS veya eşleşen çizgi öneriyi koşullar.
Diğer ölçümler ağırlık hesabında işlenir.
"""

import math

import numpy as np

from estimation.fastslam1 import (
    FastSLAM1Backend,
    log_gaussian,
)
from estimation.line_ekf_slam import (
    LineEKFSLAM,
    line_observation,
)
from estimation.pose_ekf import motion_prediction
from perception.line_association import (
    associate_lines,
    transform_line,
)
from perception.line_features import extract_lines
from simulator.robot import Pose, wrap_angle


def conditioned_gaussian(
    mean,
    prior,
    residual,
    jacobian,
    noise,
):
    """Yerel lineer ölçümle Gaussian öneri oluşturur."""
    innovation = (
        jacobian @ prior @ jacobian.T
        + noise
    )

    gain = np.linalg.solve(
        innovation,
        (prior @ jacobian.T).T,
    ).T

    proposal_mean = mean + gain @ residual
    proposal_mean[2] = wrap_angle(
        proposal_mean[2]
    )

    a = (
        np.eye(len(mean))
        - gain @ jacobian
    )

    covariance = (
        a @ prior @ a.T
        + gain @ noise @ gain.T
    )
    covariance = (
        covariance + covariance.T
    ) / 2.0

    np.linalg.cholesky(covariance)

    return proposal_mean, covariance


def state_difference(first, second):
    difference = first - second
    difference[2] = wrap_angle(
        difference[2]
    )
    return difference


class FastSLAM2Backend(FastSLAM1Backend):
    name = "fastslam2_line"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = None
        self.proposal_history = []

    def predict(self, encoder, gyro):
        dt = encoder.dt_s

        if not (
            math.isclose(
                encoder.time_s,
                self.time_s + dt,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                encoder.time_s,
                gyro.time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                dt,
                gyro.dt_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "Hareket zamanları eşleşmiyor."
            )

        distance = (
            2.0 * math.pi / 2048.0
        ) * (
            encoder.left_delta_counts
            * self.geometry.left_radius
            + encoder.right_delta_counts
            * self.geometry.right_radius
        ) / 2.0

        self.pending = []

        for particle in self.particles:
            mean, _, control = motion_prediction(
                particle.state,
                distance,
                gyro.yaw_rate_rad_s * dt,
                dt,
            )

            prior = (
                control
                @ np.diag([
                    0.005 ** 2 * dt,
                    gyro.nominal_variance * dt ** 2,
                ])
                @ control.T
            )

            prior[3, 3] += (
                0.001 ** 2 * dt
            )

            # Yerel Gaussian modelin açıklanan
            # pozitif tanımlı kovaryans tabanı.
            prior += np.eye(4) * 1e-12

            self.pending.append((
                mean.copy(),
                prior,
            ))

            particle.state = (
                self.rng.multivariate_normal(
                    mean,
                    prior,
                )
            )
            particle.state[2] = wrap_angle(
                particle.state[2]
            )

        self.time_s = encoder.time_s

    def _propose(self, kind, measurement_terms):
        if self.pending is None:
            return

        used = 0
        prior_traces = []
        proposal_traces = []
        corrections = []

        pairs = zip(
            self.particles,
            self.pending,
        )

        for index, (
            particle,
            (mean, prior),
        ) in enumerate(pairs):
            terms = measurement_terms(
                particle,
                mean,
            )

            if terms is None:
                # Mevcut örnek hareket öncelinden
                # çekilmiş durumda kalır.
                continue

            residual, jacobian, noise = terms

            proposal_mean, proposal_cov = (
                conditioned_gaussian(
                    mean,
                    prior,
                    residual,
                    jacobian,
                    noise,
                )
            )

            sample = (
                self.rng.multivariate_normal(
                    proposal_mean,
                    proposal_cov,
                )
            )
            sample[2] = wrap_angle(sample[2])

            correction = (
                log_gaussian(
                    state_difference(
                        sample,
                        mean,
                    ),
                    prior,
                )
                - log_gaussian(
                    state_difference(
                        sample,
                        proposal_mean,
                    ),
                    proposal_cov,
                )
            )

            particle.state = sample
            self.log_weights[index] += correction

            # Ölçüm olabilirliği burada eklenmez.
            # Temel sınıf onu yalnız bir kez ekler.
            used += 1

            prior_traces.append(
                float(np.trace(prior))
            )
            proposal_traces.append(
                float(np.trace(proposal_cov))
            )
            corrections.append(correction)

        self.proposal_history.append({
            "time_s": self.time_s,
            "source": kind,
            "conditioned_particles": used,
            "mean_prior_trace": (
                sum(prior_traces) / used
                if used
                else None
            ),
            "mean_proposal_trace": (
                sum(proposal_traces) / used
                if used
                else None
            ),
            "mean_log_p_over_q": (
                sum(corrections) / used
                if used
                else None
            ),
        })

        # Aynı zamandaki ikinci sensör aynı
        # hareket öncelini yeniden kullanmaz.
        self.pending = None

    def update_gps(self, packet):
        if (
            not math.isclose(
                packet.time_s,
                self.time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or (
                self._last_gps_time is not None
                and packet.time_s
                <= self._last_gps_time
            )
        ):
            raise ValueError(
                "Geçersiz GPS zamanı."
            )

        if packet.available:
            def terms(particle, mean):
                jacobian = np.zeros((2, 4))
                jacobian[:, :2] = np.eye(2)

                noise = np.eye(2) * max(
                    packet.nominal_variance_m2,
                    1e-8,
                )

                residual = np.array([
                    packet.x_m,
                    packet.y_m,
                ]) - mean[:2]

                return (
                    residual,
                    jacobian,
                    noise,
                )

            self._propose("gps", terms)

        return super().update_gps(packet)

    def update_scan(self, sensor_id, points):
        if sensor_id not in self.mounts:
            raise ValueError(
                "Montajı tanımsız LiDAR."
            )

        mount = self.mounts[sensor_id]
        features = extract_lines(points)

        def terms(particle, mean):
            pose = Pose(*mean[:3])

            world = [
                transform_line(
                    feature,
                    mount.world_pose(pose),
                )
                for feature in features
            ]

            references = [
                landmark.feature()
                for landmark in particle.landmarks
            ]

            matched = [
                row
                for row in associate_lines(
                    world,
                    references,
                )
                if row["status"] == "matched"
            ]

            if not matched:
                return None

            match = min(
                matched,
                key=lambda row: row["score"],
            )

            landmark = particle.landmarks[
                match["reference_index"]
            ]

            measurement = (
                LineEKFSLAM._measurement(
                    features[
                        match["observation_index"]
                    ]
                )
            )

            (
                expected,
                robot_jacobian,
                map_jacobian,
            ) = line_observation(
                mean,
                landmark.mean,
                mount,
            )

            if abs(wrap_angle(
                measurement[1] - expected[1]
            )) > math.pi / 2.0:
                measurement[0] *= -1.0
                measurement[1] = wrap_angle(
                    measurement[1] + math.pi
                )

            residual = measurement - expected
            residual[1] = wrap_angle(
                residual[1]
            )

            noise = (
                map_jacobian
                @ landmark.covariance
                @ map_jacobian.T
                + self.R
            )

            return (
                residual,
                robot_jacobian,
                noise,
            )

        self._propose(
            f"lidar/{sensor_id}",
            terms,
        )

        super().update_scan(
            sensor_id,
            points,
        )