"""Kimliği korunan, yapılandırılabilir çoklu LiDAR simülasyonu."""

from dataclasses import dataclass, field
import math

from simulator.environment import Environment
from simulator.lidar import Lidar, LidarConfig, LidarScan
from simulator.mounting import SensorMount
from simulator.robot import Pose


@dataclass(frozen=True)
class LidarMeasurement:
    """
    Algoritmalara aktarılabilecek ölçüm paketi.

    Gerçek robot pozu veya gerçek montaj bilgisi içermez.
    Tarama noktaları sensör koordinatlarındadır.
    """

    sensor_id: str
    time_s: float
    scan: LidarScan


@dataclass(frozen=True)
class LidarUnit:
    """Simülatördeki bir fiziksel LiDAR."""

    sensor_id: str
    actual_mount: SensorMount = field(
        default_factory=SensorMount
    )
    config: LidarConfig = field(
        default_factory=LidarConfig
    )

    def __post_init__(self):
        if (
            not isinstance(self.sensor_id, str)
            or not self.sensor_id.strip()
        ):
            raise ValueError(
                "Sensör kimliği boş olmayan bir metin olmalıdır."
            )

        if self.sensor_id != self.sensor_id.strip():
            raise ValueError(
                "Sensör kimliğinin başında veya sonunda "
                "boşluk olmamalıdır."
            )

    def measure(
        self,
        robot_pose: Pose,
        environment: Environment,
        time_s: float,
    ) -> LidarMeasurement:
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError(
                "Ölçüm zamanı sonlu ve negatif olmayan "
                "bir değer olmalıdır."
            )

        sensor_pose = self.actual_mount.world_pose(
            robot_pose
        )

        scan = Lidar(self.config).scan(
            sensor_pose,
            environment,
        )

        return LidarMeasurement(
            sensor_id=self.sensor_id,
            time_s=time_s,
            scan=scan,
        )


def scan_all(
    units: tuple[LidarUnit, ...],
    robot_pose: Pose,
    environment: Environment,
    time_s: float,
) -> tuple[LidarMeasurement, ...]:
    """Sensörleri aynı robot pozunda ve zamanda tarar."""
    if not units:
        raise ValueError(
            "En az bir LiDAR tanımlanmalıdır."
        )

    sensor_ids = [unit.sensor_id for unit in units]

    if len(sensor_ids) != len(set(sensor_ids)):
        raise ValueError(
            "Her LiDAR'ın kimliği farklı olmalıdır."
        )

    return tuple(
        unit.measure(
            robot_pose,
            environment,
            time_s,
        )
        for unit in units
    )