"""İdeal diferansiyel sürüşlü robotun 2B hareket modeli."""

from dataclasses import dataclass
import math


def wrap_angle(angle: float) -> float:
    """Açıyı [-π, π) aralığına getirir."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose:
    """Robotun dünya koordinatlarındaki konumu ve yönü."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def __post_init__(self):
        values = (self.x, self.y, self.theta)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Poz değerleri sonlu olmalıdır.")

        object.__setattr__(self, "theta", wrap_angle(self.theta))


@dataclass(frozen=True)
class RobotGeometry:
    """Uzunluklar metre cinsindedir."""

    left_radius: float = 0.05
    right_radius: float = 0.05
    wheelbase: float = 0.30

    def __post_init__(self):
        values = (
            self.left_radius,
            self.right_radius,
            self.wheelbase,
        )

        if not all(
            math.isfinite(value) and value > 0.0
            for value in values
        ):
            raise ValueError(
                "Tekerlek yarıçapları ve aks açıklığı "
                "pozitif ve sonlu olmalıdır."
            )


class DifferentialDriveRobot:
    """Kayma ve ölçüm gürültüsü içermeyen hareket modeli."""

    def __init__(
        self,
        geometry: RobotGeometry | None = None,
        initial_pose: Pose | None = None,
    ):
        self.geometry = (
            geometry if geometry is not None else RobotGeometry()
        )
        self.pose = (
            initial_pose if initial_pose is not None else Pose()
        )

    def step(
        self,
        left_rate: float,
        right_rate: float,
        dt: float,
    ) -> Pose:
        """
        Tekerlek hızları: rad/s.
        dt: saniye.
        Hızlar bu zaman aralığında sabit kabul edilir.
        """
        values = (left_rate, right_rate, dt)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Hızlar ve süre sonlu olmalıdır.")

        if dt < 0.0:
            raise ValueError("Süre negatif olamaz.")

        # Tekerleklerin doğrusal hızları: m/s.
        left_velocity = self.geometry.left_radius * left_rate
        right_velocity = self.geometry.right_radius * right_rate

        # Robot merkezinin doğrusal ve açısal hızı.
        velocity = (left_velocity + right_velocity) / 2.0
        yaw_rate = (
            right_velocity - left_velocity
        ) / self.geometry.wheelbase

        # Sabit hız altında tam yay integrasyonu.
        half_turn = yaw_rate * dt / 2.0

        if abs(half_turn) < 1e-6:
            sinc = (
                1.0
                - half_turn**2 / 6.0
                + half_turn**4 / 120.0
            )
        else:
            sinc = math.sin(half_turn) / half_turn

        distance = velocity * dt * sinc
        direction = self.pose.theta + half_turn

        self.pose = Pose(
            x=self.pose.x + distance * math.cos(direction),
            y=self.pose.y + distance * math.sin(direction),
            theta=self.pose.theta + yaw_rate * dt,
        )

        return self.pose