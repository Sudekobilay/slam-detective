"""Yalnız gyro ölçümlerini entegre ederek yön hesaplar."""

import math

from simulator.imu import GyroMeasurement
from simulator.robot import wrap_angle


class GyroHeading:

    def __init__(self, initial_heading: float = 0.0):
        if not math.isfinite(initial_heading):
            raise ValueError("Başlangıç yönü sonlu olmalıdır.")

        self.theta = wrap_angle(initial_heading)
        self.time_s = 0.0

    def update(self, packet: GyroMeasurement) -> float:
        expected_time = self.time_s + packet.dt_s

        if not math.isclose(
            packet.time_s,
            expected_time,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Gyro paketleri sıralı ve kesintisiz olmalıdır."
            )

        self.theta = wrap_angle(
            self.theta
            + packet.yaw_rate_rad_s * packet.dt_s
        )
        self.time_s = packet.time_s

        return self.theta