"""Yerel 2B koordinatlarda GPS konum ölçümü."""

from dataclasses import dataclass
import math
import random

from simulator.robot import Pose


@dataclass(frozen=True)
class GPSConfig:
    noise_std_m: float = 0.0
    bias_x_m: float = 0.0
    bias_y_m: float = 0.0

    # Kesinti aralığı [başlangıç, bitiş).
    outage_start_s: float | None = None
    outage_end_s: float | None = None

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
            self.noise_std_m,
            self.bias_x_m,
            self.bias_y_m,
        )):
            raise ValueError("GPS parametreleri sonlu olmalıdır.")

        if self.noise_std_m < 0.0:
            raise ValueError("Standart sapma negatif olamaz.")

        start = self.outage_start_s
        end = self.outage_end_s

        if (start is None) != (end is None):
            raise ValueError(
                "Kesinti başlangıcı ve bitişi birlikte verilmelidir."
            )

        if start is not None:
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and 0.0 <= start < end
            ):
                raise ValueError("Geçersiz GPS kesinti aralığı.")


@dataclass(frozen=True)
class GPSMeasurement:
    time_s: float
    available: bool
    x_m: float | None
    y_m: float | None
    nominal_variance_m2: float

    def __post_init__(self):
        if not (
            math.isfinite(self.time_s)
            and self.time_s >= 0.0
            and math.isfinite(self.nominal_variance_m2)
            and self.nominal_variance_m2 >= 0.0
        ):
            raise ValueError("Geçersiz GPS zamanı veya varyansı.")

        if self.available:
            if (
                self.x_m is None
                or self.y_m is None
                or not math.isfinite(self.x_m)
                or not math.isfinite(self.y_m)
            ):
                raise ValueError("Geçerli konum ölçümü gerekli.")
        elif self.x_m is not None or self.y_m is not None:
            raise ValueError(
                "Kesintide konum alanları None olmalıdır."
            )


class GPSSensor:

    def __init__(
        self,
        config: GPSConfig | None = None,
        seed: int = 101,
    ):
        self.config = (
            config if config is not None else GPSConfig()
        )
        self._rng = random.Random(seed)
        self._last_time = None

    def measure(
        self,
        true_pose: Pose,
        time_s: float,
    ) -> GPSMeasurement:
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("Geçersiz ölçüm zamanı.")

        if self._last_time is not None:
            if time_s <= self._last_time:
                raise ValueError(
                    "GPS ölçüm zamanları artan sırada olmalıdır."
                )

        # Kesinti durumunda da örnekleri çekiyoruz:
        # eşleştirilmiş deneylerde gürültü akışı kaymasın.
        noise_x = self._rng.gauss(0.0, self.config.noise_std_m)
        noise_y = self._rng.gauss(0.0, self.config.noise_std_m)

        start = self.config.outage_start_s
        end = self.config.outage_end_s

        available = not (
            start is not None and start <= time_s < end
        )

        packet = GPSMeasurement(
            time_s=time_s,
            available=available,
            x_m=(
                true_pose.x + self.config.bias_x_m + noise_x
                if available else None
            ),
            y_m=(
                true_pose.y + self.config.bias_y_m + noise_y
                if available else None
            ),
            nominal_variance_m2=self.config.noise_std_m ** 2,
        )

        self._last_time = time_s
        return packet