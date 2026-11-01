"""Sonlu duvar parçalarına ışın göndererek çalışan ideal 2B LiDAR."""

from dataclasses import dataclass
import math

from simulator.environment import Environment, Point, Wall
from simulator.robot import Pose


def cross(first: Point, second: Point) -> float:
    """İki boyutlu vektörlerin çapraz çarpımının skaler değeri."""
    return first[0] * second[1] - first[1] * second[0]


def ray_wall_distance(
    origin: Point,
    direction: Point,
    wall: Wall,
) -> float | None:
    """
    Birim uzunluklu ışın yönü için duvara olan mesafeyi bulur.

    Kesişim ışının gerisindeyse veya duvar parçasının
    dışındaysa None döndürür.
    """
    segment = (
        wall.end[0] - wall.start[0],
        wall.end[1] - wall.start[1],
    )
    offset = (
        wall.start[0] - origin[0],
        wall.start[1] - origin[1],
    )

    denominator = cross(direction, segment)
    parallel_tolerance = 1e-12 * wall.length

    if abs(denominator) <= parallel_tolerance:
        # Paralel fakat aynı doğru üzerinde olmayan çizgiler.
        if abs(cross(offset, direction)) > 1e-10:
            return None

        # Aynı doğru üzerindeki duvar uçlarını ışına izdüşür.
        projections = [
            (point[0] - origin[0]) * direction[0]
            + (point[1] - origin[1]) * direction[1]
            for point in (wall.start, wall.end)
        ]

        low = min(projections)
        high = max(projections)

        if high < 0.0:
            return None

        if low <= 0.0 <= high:
            return 0.0

        return low

    distance = cross(offset, segment) / denominator
    wall_fraction = cross(offset, direction) / denominator

    if (
        distance >= -1e-10
        and -1e-10 <= wall_fraction <= 1.0 + 1e-10
    ):
        return max(0.0, distance)

    return None


@dataclass(frozen=True)
class LidarConfig:
    """Açılar radyan, menzil metre cinsindedir."""

    field_of_view: float = 2.0 * math.pi
    beam_count: int = 360
    max_range: float = 15.0

    def __post_init__(self):
        if not (
            math.isfinite(self.field_of_view)
            and 0.0 < self.field_of_view <= 2.0 * math.pi
        ):
            raise ValueError(
                "Görüş açısı (0, 2π] aralığında olmalıdır."
            )

        if type(self.beam_count) is not int or self.beam_count < 1:
            raise ValueError(
                "Işın sayısı pozitif tam sayı olmalıdır."
            )

        if not (
            math.isfinite(self.max_range)
            and self.max_range > 0.0
        ):
            raise ValueError(
                "Maksimum menzil pozitif ve sonlu olmalıdır."
            )


@dataclass(frozen=True)
class LidarScan:
    """
    Sensör koordinatlarında tarama.

    None: Bu yönde menzil içinde yüzey bulunamadı.
    """
    angles: tuple[float, ...]
    ranges: tuple[float | None, ...]
    points: tuple[Point | None, ...]


class Lidar:

    def __init__(self, config: LidarConfig | None = None):
        self.config = (
            config if config is not None else LidarConfig()
        )

    def beam_angles(self) -> tuple[float, ...]:
        """Sensörün ileri yönüne göre ışın açılarını üretir."""
        count = self.config.beam_count
        fov = self.config.field_of_view

        if count == 1:
            return (0.0,)

        full_circle = abs(fov - 2.0 * math.pi) < 1e-12

        # 360° taramada -π ve +π aynı yöndür.
        # Son noktayı tekrarlamamak için bölen count olur.
        divisor = count if full_circle else count - 1

        return tuple(
            -fov / 2.0 + index * fov / divisor
            for index in range(count)
        )

    def scan(
        self,
        sensor_pose: Pose,
        environment: Environment,
    ) -> LidarScan:
        angles = self.beam_angles()
        ranges = []
        points = []

        origin = (sensor_pose.x, sensor_pose.y)
        walls = environment.walls

        for local_angle in angles:
            world_angle = sensor_pose.theta + local_angle
            direction = (
                math.cos(world_angle),
                math.sin(world_angle),
            )

            nearest = math.inf

            for wall in walls:
                distance = ray_wall_distance(
                    origin, direction, wall
                )

                if distance is not None:
                    nearest = min(nearest, distance)

            if nearest <= self.config.max_range:
                ranges.append(nearest)

                # Algoritmalara verilecek nokta sensör
                # koordinatlarında tutulur.
                points.append((
                    nearest * math.cos(local_angle),
                    nearest * math.sin(local_angle),
                ))
            else:
                ranges.append(None)
                points.append(None)

        return LidarScan(
            angles=angles,
            ranges=tuple(ranges),
            points=tuple(points),
        )