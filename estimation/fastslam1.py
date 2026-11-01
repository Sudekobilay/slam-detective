"""FastSLAM 1.0 çekirdeğinin çizgi özelliklerine uyarlaması.

Hareket öncelinden örnekleme yapılır.
Her parçacık bağımsız çizgi EKF'leri taşır.

Veri eşleştirme açgözlüdür. Yeni/belirsiz özellik
ağırlığı sezgiseldir. Dengeli ağaç ve logaritmik
karmaşıklık iddiası yoktur.
"""

import copy
from dataclasses import dataclass, replace
import math

import numpy as np

from estimation.line_ekf_slam import (
    LineEKFSLAM,
    line_initialization,
    line_observation,
)
from estimation.pose_ekf import motion_prediction
from perception.line_association import (
    associate_lines,
    transform_line,
)
from perception.line_features import extract_lines
from simulator.robot import (
    Pose,
    RobotGeometry,
    wrap_angle,
)


@dataclass
class LineLandmark:
    mean: np.ndarray
    covariance: np.ndarray
    anchor: object

    def feature(self):
        rho, alpha = self.mean

        normal = np.array([
            math.cos(alpha),
            math.sin(alpha),
        ])

        def project(point):
            point = np.asarray(point)
            projected = (
                point
                + (rho - normal @ point) * normal
            )
            return tuple(
                float(value)
                for value in projected
            )

        start = project(self.anchor.start)
        end = project(self.anchor.end)

        return replace(
            self.anchor,
            normal=tuple(
                float(value)
                for value in normal
            ),
            rho_m=float(rho),
            start=start,
            end=end,
            length_m=math.dist(start, end),
        )


@dataclass
class Particle:
    state: np.ndarray
    landmarks: list


def log_gaussian(residual, covariance):
    lower = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(
        lower,
        residual,
    )

    return float(
        -0.5 * (
            whitened @ whitened
            + 2.0 * np.log(
                np.diag(lower)
            ).sum()
            + len(residual) * math.log(
                2.0 * math.pi
            )
        )
    )


class FastSLAM1Backend:
    name = "fastslam1_line"

    def __init__(
        self,
        mounts,
        particle_count=40,
        seed=901,
    ):
        if (
            type(particle_count) is not int
            or particle_count < 2
        ):
            raise ValueError(
                "En az iki parçacık gerekli."
            )

        self.mounts = dict(mounts)
        self.geometry = RobotGeometry()
        self.rng = np.random.default_rng(seed)
        self.count = particle_count

        initial = np.array([
            1.0, 3.0, 0.0, 0.0
        ])

        # EKF'nin başlangıç kovaryansıyla
        # aynı nominal standart sapmalar.
        std = np.array([
            0.001, 0.001, 0.005, 0.02
        ])

        self.particles = [
            Particle(
                initial
                + self.rng.normal(size=4) * std,
                [],
            )
            for _ in range(particle_count)
        ]

        self.log_weights = np.full(
            particle_count,
            -math.log(particle_count),
        )

        self.R = np.diag([
            0.02 ** 2,
            math.radians(1.0) ** 2,
        ])

        self.time_s = 0.0
        self._last_gps_time = None
        self.gps_updates = 0
        self.gps_missing = 0
        self.resample_count = 0
        self.history = []

        # Ortak deney arayüzüne uyum.
        self.estimator = self

    @property
    def weights(self):
        shifted = (
            self.log_weights
            - np.max(self.log_weights)
        )
        values = np.exp(shifted)
        return values / values.sum()

    @property
    def pose(self):
        states = np.asarray([
            particle.state
            for particle in self.particles
        ])

        weights = self.weights
        xy = weights @ states[:, :2]

        theta = math.atan2(
            float(
                weights @ np.sin(states[:, 2])
            ),
            float(
                weights @ np.cos(states[:, 2])
            ),
        )

        return Pose(
            float(xy[0]),
            float(xy[1]),
            theta,
        )

    @property
    def landmark_count(self):
        best_index = int(
            np.argmax(self.weights)
        )
        return len(
            self.particles[
                best_index
            ].landmarks
        )

    def map_features(self):
        # Parçacıkların farklı harita kimlikleri
        # birbirleriyle ortalanmaz.
        best_index = int(
            np.argmax(self.weights)
        )
        best = self.particles[best_index]

        return [
            landmark.feature()
            for landmark in best.landmarks
        ]

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
                gyro.time_s,
                encoder.time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                gyro.dt_s,
                dt,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "Hareket paketleri zamanla eşleşmiyor."
            )

        radians_per_count = (
            2.0 * math.pi / 2048.0
        )

        distance = radians_per_count * (
            encoder.left_delta_counts
            * self.geometry.left_radius
            + encoder.right_delta_counts
            * self.geometry.right_radius
        ) / 2.0

        for particle in self.particles:
            noisy_distance = (
                distance
                + self.rng.normal(
                    0.0,
                    0.005 * math.sqrt(dt),
                )
            )

            noisy_turn = (
                gyro.yaw_rate_rad_s * dt
                + self.rng.normal(
                    0.0,
                    math.sqrt(
                        gyro.nominal_variance
                    ) * dt,
                )
            )

            particle.state = motion_prediction(
                particle.state,
                noisy_distance,
                noisy_turn,
                dt,
            )[0]

            particle.state[3] += (
                self.rng.normal(
                    0.0,
                    0.001 * math.sqrt(dt),
                )
            )

        self.time_s = encoder.time_s

    def _resample_if_needed(self, kind):
        weights = self.weights

        ess = float(
            1.0 / (weights @ weights)
        )
        resampled = ess < self.count / 2.0

        if resampled:
            positions = (
                self.rng.random()
                + np.arange(self.count)
            ) / self.count

            cumulative = np.cumsum(weights)
            cumulative[-1] = 1.0

            indices = np.searchsorted(
                cumulative,
                positions,
                side="right",
            )

            self.particles = [
                copy.deepcopy(
                    self.particles[int(index)]
                )
                for index in indices
            ]

            self.log_weights[:] = (
                -math.log(self.count)
            )

            self.resample_count += 1

        else:
            maximum = float(
                np.max(self.log_weights)
            )

            self.log_weights -= (
                maximum
                + math.log(float(
                    np.exp(
                        self.log_weights - maximum
                    ).sum()
                ))
            )

        self.history.append({
            "time_s": self.time_s,
            "sensor": kind,
            "ess_before_resampling": ess,
            "resampled": bool(resampled),
        })

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

        self._last_gps_time = packet.time_s

        if not packet.available:
            self.gps_missing += 1
            return False

        covariance = np.eye(2) * max(
            packet.nominal_variance_m2,
            1e-8,
        )

        measurement = np.array([
            packet.x_m,
            packet.y_m,
        ])

        for index, particle in enumerate(
            self.particles
        ):
            self.log_weights[index] += (
                log_gaussian(
                    measurement
                    - particle.state[:2],
                    covariance,
                )
            )

        self._resample_if_needed("gps")
        self.gps_updates += 1

        return True

    def update_scan(self, sensor_id, points):
        if sensor_id not in self.mounts:
            raise ValueError(
                "Montajı tanımsız LiDAR."
            )

        mount = self.mounts[sensor_id]
        features = extract_lines(points)

        for index, particle in enumerate(
            self.particles
        ):
            pose = Pose(
                *particle.state[:3]
            )

            world = [
                transform_line(
                    feature,
                    mount.world_pose(pose),
                )
                for feature in features
            ]

            references = [
                landmark.feature()
                for landmark
                in particle.landmarks
            ]

            matches = associate_lines(
                world,
                references,
            )

            for match in matches:
                feature = features[
                    match["observation_index"]
                ]

                measurement = (
                    LineEKFSLAM._measurement(
                        feature
                    )
                )

                if match["status"] == "matched":
                    landmark = (
                        particle.landmarks[
                            match["reference_index"]
                        ]
                    )

                    (
                        expected,
                        _,
                        jacobian,
                    ) = line_observation(
                        particle.state,
                        landmark.mean,
                        mount,
                    )

                    angle_error = wrap_angle(
                        measurement[1]
                        - expected[1]
                    )

                    if (
                        abs(angle_error)
                        > math.pi / 2.0
                    ):
                        measurement[0] *= -1.0
                        measurement[1] = wrap_angle(
                            measurement[1]
                            + math.pi
                        )

                    residual = (
                        measurement - expected
                    )
                    residual[1] = wrap_angle(
                        residual[1]
                    )

                    innovation = (
                        jacobian
                        @ landmark.covariance
                        @ jacobian.T
                        + self.R
                    )

                    self.log_weights[index] += (
                        log_gaussian(
                            residual,
                            innovation,
                        )
                    )

                    gain = np.linalg.solve(
                        innovation,
                        (
                            landmark.covariance
                            @ jacobian.T
                        ).T,
                    ).T

                    landmark.mean += (
                        gain @ residual
                    )
                    landmark.mean[1] = wrap_angle(
                        landmark.mean[1]
                    )

                    a = (
                        np.eye(2)
                        - gain @ jacobian
                    )

                    landmark.covariance = (
                        a
                        @ landmark.covariance
                        @ a.T
                        + gain @ self.R @ gain.T
                    )

                    landmark.covariance = (
                        landmark.covariance
                        + landmark.covariance.T
                    ) / 2.0

                else:
                    # Sezgisel yeni/aykırı
                    # özellik modeli.
                    self.log_weights[index] -= (
                        math.log(
                            40.0 * math.pi
                        )
                    )

                    if (
                        match["status"]
                        == "no_candidate"
                    ):
                        (
                            mean,
                            _,
                            jacobian,
                        ) = line_initialization(
                            particle.state,
                            measurement,
                            mount,
                        )

                        covariance = (
                            jacobian
                            @ self.R
                            @ jacobian.T
                        )

                        particle.landmarks.append(
                            LineLandmark(
                                mean,
                                covariance,
                                world[
                                    match[
                                        "observation_index"
                                    ]
                                ],
                            )
                        )

        self._resample_if_needed(sensor_id)

    def snapshot(self, time_s):
        pose = self.pose

        return {
            "time_s": time_s,
            "x_m": pose.x,
            "y_m": pose.y,
            "theta_rad": pose.theta,
        }