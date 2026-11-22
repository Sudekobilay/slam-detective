"""Shared SE(2) values; no simulator state."""
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
