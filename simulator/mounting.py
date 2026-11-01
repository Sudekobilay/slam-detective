"""Sensör, robot ve dünya koordinatları arasındaki dönüşümler."""

from dataclasses import dataclass
import math

from simulator.environment import Point
from simulator.robot import Pose, wrap_angle


def transform_point(point: Point, frame_pose: Pose) -> Point:
    """
    Yerel koordinattaki noktayı üst koordinat sistemine taşır.

    Önce döndürme, ardından öteleme uygulanır.
    """
    if len(point) != 2:
        raise ValueError("Nokta iki koordinat içermelidir.")

    if not all(math.isfinite(value) for value in point):
        raise ValueError("Nokta koordinatları sonlu olmalıdır.")

    cosine = math.cos(frame_pose.theta)
    sine = math.sin(frame_pose.theta)

    return (
        frame_pose.x + cosine * point[0] - sine * point[1],
        frame_pose.y + sine * point[0] + cosine * point[1],
    )


@dataclass(frozen=True)
class SensorMount:
    """
    Sensörün robot merkezine göre konumu ve yönü.

    x: ileri yöndeki uzaklık (m).
    y: robotun soluna doğru uzaklık (m).
    yaw: robotun ileri yönüne göre açı (rad).
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def __post_init__(self):
        values = (self.x, self.y, self.yaw)

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "Montaj değerleri sonlu olmalıdır."
            )

        object.__setattr__(
            self, "yaw", wrap_angle(self.yaw)
        )

    def world_pose(self, robot_pose: Pose) -> Pose:
        """Robot pozu ve montajdan sensörün dünya pozunu bulur."""
        world_x, world_y = transform_point(
            (self.x, self.y),
            robot_pose,
        )

        return Pose(
            x=world_x,
            y=world_y,
            theta=robot_pose.theta + self.yaw,
        )

    def point_to_robot(self, sensor_point: Point) -> Point:
        """Sensör koordinatındaki noktayı robot koordinatına taşır."""
        return transform_point(
            sensor_point,
            Pose(self.x, self.y, self.yaw),
        )

    def point_to_world(
        self,
        sensor_point: Point,
        robot_pose: Pose,
    ) -> Point:
        """Sensör noktasını robot üzerinden dünya koordinatına taşır."""
        robot_point = self.point_to_robot(sensor_point)

        return transform_point(robot_point, robot_pose)