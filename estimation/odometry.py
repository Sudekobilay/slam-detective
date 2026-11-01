"""Encoder ölçümlerinden robot pozu tahmini."""

import math

from simulator.encoder import EncoderConfig, EncoderMeasurement
from simulator.robot import (
    DifferentialDriveRobot,
    Pose,
    RobotGeometry,
)


class WheelOdometry:

    def __init__(
        self,
        geometry: RobotGeometry | None = None,
        encoder_config: EncoderConfig | None = None,
        initial_pose: Pose | None = None,
    ):
        self.encoder_config = (
            encoder_config
            if encoder_config is not None
            else EncoderConfig()
        )

        # Buradaki geometri, tahmin algoritmasının
        # bildiği kalibrasyon değerleridir.
        self._model = DifferentialDriveRobot(
            geometry=geometry,
            initial_pose=initial_pose,
        )
        self.time_s = 0.0

    @property
    def pose(self) -> Pose:
        return self._model.pose

    def update(self, measurement: EncoderMeasurement) -> Pose:
        expected_time = self.time_s + measurement.dt_s

        if not math.isclose(
            measurement.time_s,
            expected_time,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Encoder paketleri sıralı ve kesintisiz olmalıdır."
            )

        radians_per_count = (
            2.0 * math.pi
            / self.encoder_config.counts_per_revolution
        )

        left_rate = (
            measurement.left_delta_counts
            * radians_per_count
            / measurement.dt_s
        )
        right_rate = (
            measurement.right_delta_counts
            * radians_per_count
            / measurement.dt_s
        )

        pose = self._model.step(
            left_rate,
            right_rate,
            measurement.dt_s,
        )

        self.time_s = measurement.time_s
        return pose