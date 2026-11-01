"""Eşzamanlı encoder ve gyro ölçümlerinden hareket tahmini."""

import math

from simulator.encoder import EncoderConfig, EncoderMeasurement
from simulator.imu import GyroMeasurement
from simulator.robot import (
    DifferentialDriveRobot,
    Pose,
    RobotGeometry,
)


class EncoderGyroOdometry:

    def __init__(
        self,
        gyro_weight: float = 0.5,
        initial_pose: Pose | None = None,
        geometry: RobotGeometry | None = None,
        encoder_config: EncoderConfig | None = None,
    ):
        if not (
            math.isfinite(gyro_weight)
            and 0.0 <= gyro_weight <= 1.0
        ):
            raise ValueError("Gyro ağırlığı [0,1] içinde olmalıdır.")

        self.gyro_weight = gyro_weight
        self.geometry = (
            geometry if geometry is not None else RobotGeometry()
        )
        self.encoder_config = (
            encoder_config
            if encoder_config is not None
            else EncoderConfig()
        )
        self._robot = DifferentialDriveRobot(
            geometry=self.geometry,
            initial_pose=initial_pose,
        )
        self.time_s = 0.0

    @property
    def pose(self) -> Pose:
        return self._robot.pose

    def update(
        self,
        encoder: EncoderMeasurement,
        gyro: GyroMeasurement,
    ) -> Pose:
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
            raise ValueError(
                "Ölçümler aynı aralığa ait, sıralı "
                "ve kesintisiz olmalıdır."
            )

        radians_per_count = (
            2.0 * math.pi
            / self.encoder_config.counts_per_revolution
        )

        left_distance = (
            encoder.left_delta_counts
            * radians_per_count
            * self.geometry.left_radius
        )
        right_distance = (
            encoder.right_delta_counts
            * radians_per_count
            * self.geometry.right_radius
        )

        ds = (left_distance + right_distance) / 2.0
        encoder_turn = (
            right_distance - left_distance
        ) / self.geometry.wheelbase

        gyro_turn = gyro.yaw_rate_rad_s * gyro.dt_s

        combined_turn = (
            (1.0 - self.gyro_weight) * encoder_turn
            + self.gyro_weight * gyro_turn
        )

        # Birleşik ilerleme/dönüşü mevcut yay integrasyonuna ver.
        dt = encoder.dt_s
        velocity = ds / dt
        yaw_rate = combined_turn / dt

        left_rate = (
            velocity - yaw_rate * self.geometry.wheelbase / 2.0
        ) / self.geometry.left_radius
        right_rate = (
            velocity + yaw_rate * self.geometry.wheelbase / 2.0
        ) / self.geometry.right_radius

        pose = self._robot.step(left_rate, right_rate, dt)
        self.time_s = encoder.time_s

        return pose