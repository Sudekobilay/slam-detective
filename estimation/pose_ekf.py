"""Encoder, gyro ve GPS ile düzlemsel EKF konum kestirimi."""

import math
import numpy as np

from simulator.encoder import EncoderConfig, EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.robot import Pose, RobotGeometry, wrap_angle


def motion_prediction(state, ds, measured_turn, dt):
    """
    Sabit eğrilikli hareket ve analitik Jacobian'lar.

    Kontroller:
        ds: encoder ilerlemesi (m)
        measured_turn: gyro açısal hız integrali (rad)
    """
    x, y, theta, bias = state
    turn = measured_turn - bias * dt
    half = turn / 2.0

    if abs(turn) < 1e-5:
        sinc = 1.0 - turn**2 / 24.0 + turn**4 / 1920.0
        derivative = -turn / 12.0 + turn**3 / 480.0
    else:
        sinc = math.sin(half) / half
        derivative = (
            0.5
            * (half * math.cos(half) - math.sin(half))
            / half**2
        )

    direction = theta + half
    cosine = math.cos(direction)
    sine = math.sin(direction)

    result = np.array([
        x + ds * sinc * cosine,
        y + ds * sinc * sine,
        wrap_angle(theta + turn),
        bias,
    ])

    # Kontrol Jacobian'ı: sütunlar ds ve measured_turn.
    control_jacobian = np.zeros((4, 2))
    control_jacobian[:, 0] = [
        sinc * cosine,
        sinc * sine,
        0.0,
        0.0,
    ]
    control_jacobian[:, 1] = [
        ds * (derivative * cosine - 0.5 * sinc * sine),
        ds * (derivative * sine + 0.5 * sinc * cosine),
        1.0,
        0.0,
    ]

    state_jacobian = np.eye(4)
    state_jacobian[0, 2] = -ds * sinc * sine
    state_jacobian[1, 2] = ds * sinc * cosine

    # turn = measured_turn - bias * dt
    state_jacobian[:, 3] -= (
        dt * control_jacobian[:, 1]
    )

    return result, state_jacobian, control_jacobian


class PoseEKF:

    def __init__(
        self,
        initial_pose=None,
        geometry=None,
        encoder_config=None,
        initial_covariance=None,
        distance_noise_density=0.005,
        bias_random_walk=0.001,
        gps_variance_floor=1e-8,
    ):
        pose = initial_pose if initial_pose is not None else Pose()

        self.geometry = (
            geometry if geometry is not None else RobotGeometry()
        )
        self.encoder_config = (
            encoder_config
            if encoder_config is not None
            else EncoderConfig()
        )

        parameters = (
            distance_noise_density,
            bias_random_walk,
            gps_variance_floor,
        )
        if not all(math.isfinite(v) and v >= 0 for v in parameters):
            raise ValueError("Geçersiz filtre gürültü parametresi.")

        if gps_variance_floor <= 0.0:
            raise ValueError("GPS varyans alt sınırı pozitif olmalı.")

        self.distance_noise_density = distance_noise_density
        self.bias_random_walk = bias_random_walk
        self.gps_variance_floor = gps_variance_floor

        self.state = np.array([
            pose.x, pose.y, pose.theta, 0.0
        ])

        self.P = (
            np.diag([1e-6, 1e-6, 0.005**2, 0.02**2])
            if initial_covariance is None
            else np.array(initial_covariance, dtype=float, copy=True)
        )

        self.time_s = 0.0
        self._last_gps_time = None
        self.gps_updates = 0
        self.gps_missing = 0
        self.innovations = []

        self._validate_covariance(self.P)

    @staticmethod
    def _validate_covariance(matrix):
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(
                matrix, matrix.T, atol=1e-10, rtol=0.0
            )
            or np.linalg.eigvalsh(matrix).min() < -1e-10
        ):
            raise ValueError("Geçersiz kovaryans matrisi.")

    @property
    def pose(self):
        return Pose(*self.state[:3])

    def predict(
        self,
        encoder: EncoderMeasurement,
        gyro: GyroMeasurement,
    ):
        close = lambda a, b: math.isclose(
            a, b, rel_tol=0.0, abs_tol=1e-9
        )

        if not (
            close(encoder.time_s, gyro.time_s)
            and close(encoder.dt_s, gyro.dt_s)
            and close(
                encoder.time_s,
                self.time_s + encoder.dt_s,
            )
        ):
            raise ValueError("Hareket ölçümleri zamanla eşleşmiyor.")

        dt = encoder.dt_s
        angle_per_count = (
            2.0 * math.pi
            / self.encoder_config.counts_per_revolution
        )

        left_distance = (
            encoder.left_delta_counts
            * angle_per_count
            * self.geometry.left_radius
        )
        right_distance = (
            encoder.right_delta_counts
            * angle_per_count
            * self.geometry.right_radius
        )
        ds = (left_distance + right_distance) / 2.0
        measured_turn = gyro.yaw_rate_rad_s * dt

        candidate, F, G = motion_prediction(
            self.state, ds, measured_turn, dt
        )

        # Seçilmiş nominal süreç belirsizliği.
        distance_variance = (
            self.distance_noise_density**2 * dt
        )
        turn_variance = gyro.nominal_variance * dt**2

        Q = G @ np.diag([
            distance_variance, turn_variance
        ]) @ G.T
        Q[3, 3] += self.bias_random_walk**2 * dt

        covariance = F @ self.P @ F.T + Q
        covariance = (covariance + covariance.T) / 2.0

        self._validate_covariance(covariance)

        if not np.isfinite(candidate).all():
            raise ValueError("Sonlu olmayan durum tahmini.")

        self.state = candidate
        self.P = covariance
        self.time_s = encoder.time_s

    def update_gps(self, packet: GPSMeasurement):
        if not math.isclose(
            packet.time_s,
            self.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("GPS zamanı filtre zamanıyla eşleşmiyor.")

        if (
            self._last_gps_time is not None
            and packet.time_s <= self._last_gps_time
        ):
            raise ValueError("Yinelenen veya eski GPS paketi.")

        if not packet.available:
            self._last_gps_time = packet.time_s
            self.gps_missing += 1
            return False

        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        variance = max(
            packet.nominal_variance_m2,
            self.gps_variance_floor,
        )
        R = np.eye(2) * variance

        residual = np.array([
            packet.x_m, packet.y_m
        ]) - H @ self.state

        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, (self.P @ H.T).T).T

        candidate = self.state + K @ residual
        candidate[2] = wrap_angle(candidate[2])

        # Joseph biçimi: sayısal kovaryans kararlılığı.
        A = np.eye(4) - K @ H
        covariance = A @ self.P @ A.T + K @ R @ K.T
        covariance = (covariance + covariance.T) / 2.0

        self._validate_covariance(covariance)

        if not np.isfinite(candidate).all():
            raise ValueError("Sonlu olmayan GPS güncellemesi.")

        nis = float(residual @ np.linalg.solve(S, residual))

        self.state = candidate
        self.P = covariance
        self._last_gps_time = packet.time_s
        self.gps_updates += 1

        self.innovations.append({
            "time_s": packet.time_s,
            "residual_x_m": float(residual[0]),
            "residual_y_m": float(residual[1]),
            "nis": nis,
        })

        return True