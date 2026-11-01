"""Robot ve doğru haritasını ortak kovaryansla tutan EKF-SLAM çekirdeği."""

from dataclasses import replace
import math
import numpy as np

from estimation.pose_ekf import PoseEKF, motion_prediction
from perception.line_association import transform_line
from simulator.robot import Pose, wrap_angle


def line_initialization(robot_state, measurement, mount):
    """
    Sensör ölçümünden dünya doğrusu oluşturur.

    measurement = [rho_sensor, alpha_sensor]
    Dönen Jacobian'lar robot durumu ve ölçüm içindir.
    """
    pose = Pose(*robot_state[:3])
    sensor_pose = mount.world_pose(pose)

    alpha = wrap_angle(
        measurement[1] + sensor_pose.theta
    )
    normal = np.array([math.cos(alpha), math.sin(alpha)])
    normal_derivative = np.array([-normal[1], normal[0]])

    sensor_position = np.array([sensor_pose.x, sensor_pose.y])

    cosine = math.cos(pose.theta)
    sine = math.sin(pose.theta)
    rotated_offset = np.array([
        cosine * mount.x - sine * mount.y,
        sine * mount.x + cosine * mount.y,
    ])
    position_derivative = np.array([
        -rotated_offset[1], rotated_offset[0]
    ])

    landmark = np.array([
        measurement[0] + normal @ sensor_position,
        alpha,
    ])

    J_robot = np.zeros((2, 4))
    J_robot[0, :2] = normal
    J_robot[0, 2] = (
        normal_derivative @ sensor_position
        + normal @ position_derivative
    )
    J_robot[1, 2] = 1.0

    J_measurement = np.array([
        [1.0, normal_derivative @ sensor_position],
        [0.0, 1.0],
    ])

    return landmark, J_robot, J_measurement


def line_observation(robot_state, landmark, mount):
    """Dünya doğrusunun sensörde beklenen ölçümünü hesaplar."""
    pose = Pose(*robot_state[:3])
    sensor_pose = mount.world_pose(pose)

    rho, alpha = landmark
    normal = np.array([math.cos(alpha), math.sin(alpha)])
    normal_derivative = np.array([-normal[1], normal[0]])
    sensor_position = np.array([sensor_pose.x, sensor_pose.y])

    cosine = math.cos(pose.theta)
    sine = math.sin(pose.theta)
    rotated_offset = np.array([
        cosine * mount.x - sine * mount.y,
        sine * mount.x + cosine * mount.y,
    ])
    position_derivative = np.array([
        -rotated_offset[1], rotated_offset[0]
    ])

    expected = np.array([
        rho - normal @ sensor_position,
        wrap_angle(alpha - sensor_pose.theta),
    ])

    H_robot = np.zeros((2, 4))
    H_robot[0, :2] = -normal
    H_robot[0, 2] = -normal @ position_derivative
    H_robot[1, 2] = -1.0

    H_landmark = np.array([
        [1.0, -normal_derivative @ sensor_position],
        [0.0, 1.0],
    ])

    return expected, H_robot, H_landmark


class LineEKFSLAM(PoseEKF):

    def __init__(
        self,
        *args,
        line_rho_std_m=0.02,
        line_alpha_std_rad=math.radians(1.0),
        **kwargs,
    ):
        for value in (line_rho_std_m, line_alpha_std_rad):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "Doğru ölçüm belirsizlikleri pozitif olmalıdır."
                )

        initial_covariance = kwargs.get("initial_covariance")
        if initial_covariance is not None:
            if np.asarray(initial_covariance).shape != (4, 4):
                raise ValueError(
                    "Başlangıç kovaryansı yalnız robot için 4×4 olmalı."
                )

        super().__init__(*args, **kwargs)

        self.line_R = np.diag([
            line_rho_std_m**2,
            line_alpha_std_rad**2,
        ])

        self._anchors = []
        self.line_updates = 0

    @staticmethod
    def _validate_covariance(matrix):
        n = matrix.shape[0] if matrix.ndim == 2 else 0

        if (
            n < 4
            or (n - 4) % 2 != 0
            or matrix.shape != (n, n)
            or not np.isfinite(matrix).all()
            or not np.allclose(
                matrix, matrix.T, atol=1e-10, rtol=0.0
            )
            or np.linalg.eigvalsh(matrix).min() < -1e-8
        ):
            raise ValueError("Geçersiz ortak kovaryans.")

    @property
    def landmark_count(self):
        return len(self._anchors)

    def predict(self, encoder, gyro):
        """
        Doğrulanmış robot EKF tahminini kullanır;
        robot-harita çapraz kovaryanslarını da taşır.
        """
        prior_robot_state = self.state[:4].copy()

        robot_filter = PoseEKF(
            initial_pose=Pose(*prior_robot_state[:3]),
            geometry=self.geometry,
            encoder_config=self.encoder_config,
            initial_covariance=self.P[:4, :4],
            distance_noise_density=self.distance_noise_density,
            bias_random_walk=self.bias_random_walk,
            gps_variance_floor=self.gps_variance_floor,
        )
        robot_filter.state = prior_robot_state.copy()
        robot_filter.time_s = self.time_s
        robot_filter.predict(encoder, gyro)

        angle_per_count = (
            2.0 * math.pi
            / self.encoder_config.counts_per_revolution
        )
        ds = 0.5 * angle_per_count * (
            encoder.left_delta_counts * self.geometry.left_radius
            + encoder.right_delta_counts * self.geometry.right_radius
        )

        _, F_robot, _ = motion_prediction(
            prior_robot_state,
            ds,
            gyro.yaw_rate_rad_s * gyro.dt_s,
            encoder.dt_s,
        )

        F = np.eye(len(self.state))
        F[:4, :4] = F_robot

        covariance = F @ self.P @ F.T

        # Robot bloğu süreç gürültüsü dahil hesaplandı.
        covariance[:4, :4] = robot_filter.P
        covariance = (covariance + covariance.T) / 2.0

        self._validate_covariance(covariance)

        self.state[:4] = robot_filter.state
        self.P = covariance
        self.time_s = robot_filter.time_s

    @staticmethod
    def _measurement(feature):
        normal = feature.normal

        if not math.isclose(
            math.hypot(*normal), 1.0,
            rel_tol=0.0, abs_tol=1e-8,
        ):
            raise ValueError("Doğru normali birim olmalıdır.")

        z = np.array([
            feature.rho_m,
            math.atan2(normal[1], normal[0]),
        ])

        if not np.isfinite(z).all():
            raise ValueError("Sonlu doğru ölçümü gerekli.")

        return z

    def add_line(self, feature, mount):
        """Yeni doğruyu çapraz kovaryanslarıyla birlikte başlatır."""
        measurement = self._measurement(feature)

        landmark, J_robot, J_measurement = line_initialization(
            self.state[:4], measurement, mount
        )

        J = np.zeros((2, len(self.state)))
        J[:, :4] = J_robot

        cross_covariance = self.P @ J.T
        landmark_covariance = (
            J @ self.P @ J.T
            + J_measurement @ self.line_R @ J_measurement.T
        )

        covariance = np.block([
            [self.P, cross_covariance],
            [cross_covariance.T, landmark_covariance],
        ])
        covariance = (covariance + covariance.T) / 2.0
        self._validate_covariance(covariance)

        anchor = transform_line(
            feature,
            mount.world_pose(self.pose),
        )

        landmark_id = self.landmark_count

        self.state = np.concatenate([self.state, landmark])
        self.P = covariance
        self._anchors.append(anchor)

        # Başlatan ölçüm ayrıca düzeltmede tekrar kullanılmaz.
        return landmark_id

    def _correct(self, residual, H, R):
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, (self.P @ H.T).T).T

        candidate = self.state + K @ residual
        candidate[2] = wrap_angle(candidate[2])

        for index in range(5, len(candidate), 2):
            candidate[index] = wrap_angle(candidate[index])

        A = np.eye(len(self.state)) - K @ H
        covariance = A @ self.P @ A.T + K @ R @ K.T
        covariance = (covariance + covariance.T) / 2.0

        self._validate_covariance(covariance)

        if not np.isfinite(candidate).all():
            raise ValueError("Sonlu olmayan ortak durum.")

        nis = float(residual @ np.linalg.solve(S, residual))

        self.state = candidate
        self.P = covariance

        return nis

    def _line_update_terms(self, landmark_id, feature, mount):
        if (
            type(landmark_id) is not int
            or not 0 <= landmark_id < self.landmark_count
        ):
            raise ValueError("Geçersiz harita doğrusu kimliği.")

        index = 4 + 2 * landmark_id
        measurement = self._measurement(feature)

        expected, H_robot, H_landmark = line_observation(
            self.state[:4],
            self.state[index:index + 2],
            mount,
        )

        # Aynı doğrunun ters normalle gösterimini hizala.
        if abs(wrap_angle(measurement[1] - expected[1])) > math.pi / 2:
            measurement[0] *= -1.0
            measurement[1] = wrap_angle(measurement[1] + math.pi)

        residual = measurement - expected
        residual[1] = wrap_angle(residual[1])

        H = np.zeros((2, len(self.state)))
        H[:, :4] = H_robot
        H[:, index:index + 2] = H_landmark

        return measurement, expected, residual, H

    def line_innovation(self, landmark_id, feature, mount):
        """Durumu değiştirmeden, mevcut tahmine göre ölçüm farkını verir.

        Yalnız eşleştirilmiş bir harita doğrusu için tanımlıdır.
        Kovaryans, robot-harita çapraz terimlerini de içerir.
        Bu metot ölçüm reddetmez ve hata kaynağı seçmez.
        """
        measurement, expected, residual, H = self._line_update_terms(
            landmark_id, feature, mount
        )
        S = H @ self.P @ H.T + self.line_R
        nis = float(residual @ np.linalg.solve(S, residual))

        if not np.isfinite(S).all() or not math.isfinite(nis):
            raise ValueError("Sonlu olmayan doğru inovasyonu.")

        return {
            "measured_rho_m": float(measurement[0]),
            "measured_alpha_rad": float(measurement[1]),
            "expected_rho_m": float(expected[0]),
            "expected_alpha_rad": float(expected[1]),
            "residual_rho_m": float(residual[0]),
            "residual_alpha_rad": float(residual[1]),
            "innovation_covariance": S.tolist(),
            "nis": nis,
        }

    def update_line(self, landmark_id, feature, mount):
        _, _, residual, H = self._line_update_terms(
            landmark_id, feature, mount
        )
        nis = self._correct(residual, H, self.line_R)
        self.line_updates += 1
        return nis

    def update_gps(self, packet):
        if not math.isclose(
            packet.time_s, self.time_s,
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise ValueError("GPS zamanı filtreyle eşleşmiyor.")

        if (
            self._last_gps_time is not None
            and packet.time_s <= self._last_gps_time
        ):
            raise ValueError("Eski veya yinelenen GPS paketi.")

        if not packet.available:
            self._last_gps_time = packet.time_s
            self.gps_missing += 1
            return False

        H = np.zeros((2, len(self.state)))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        residual = np.array([
            packet.x_m, packet.y_m
        ]) - self.state[:2]

        R = np.eye(2) * max(
            packet.nominal_variance_m2,
            self.gps_variance_floor,
        )

        nis = self._correct(residual, H, R)

        self.innovations.append({
            "time_s": packet.time_s,
            "residual_x_m": float(residual[0]),
            "residual_y_m": float(residual[1]),
            "nis": nis,
        })
        self._last_gps_time = packet.time_s
        self.gps_updates += 1

        return True

    def map_features(self):
        """
        Harita doğrularını çizilebilir sonlu parçalar olarak döndürür.

        Uçlar ilk gözlemden kalan geometrik desteğin güncel
        doğruya izdüşümüdür; ayrı EKF değişkenleri değildir.
        """
        features = []

        for landmark_id, anchor in enumerate(self._anchors):
            index = 4 + 2 * landmark_id
            rho, alpha = self.state[index:index + 2]
            normal = np.array([
                math.cos(alpha), math.sin(alpha)
            ])

            def project(point):
                point = np.asarray(point)
                projected = point + (rho - normal @ point) * normal
                return tuple(float(value) for value in projected)

            start = project(anchor.start)
            end = project(anchor.end)

            features.append(replace(
                anchor,
                normal=tuple(float(value) for value in normal),
                rho_m=float(rho),
                start=start,
                end=end,
                length_m=math.dist(start, end),
            ))

        return features