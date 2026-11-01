"""Düzlemsel IMU'nun gyro bileşeni."""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class GyroConfig:
    # Standart sapma örnek başınadır; birim rad/s.
    noise_std_rad_s: float = 0.0
    bias_rad_s: float = 0.0

    def __post_init__(self):
        if not (
            math.isfinite(self.noise_std_rad_s)
            and self.noise_std_rad_s >= 0.0
            and math.isfinite(self.bias_rad_s)
        ):
            raise ValueError("Geçersiz gyro parametreleri.")


@dataclass(frozen=True)
class GyroMeasurement:
    time_s: float
    dt_s: float
    yaw_rate_rad_s: float
    nominal_variance: float

    def __post_init__(self):
        values = (
            self.time_s,
            self.dt_s,
            self.yaw_rate_rad_s,
            self.nominal_variance,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Ölçüm değerleri sonlu olmalıdır.")

        if (
            self.dt_s <= 0.0
            or self.time_s < self.dt_s
            or self.nominal_variance < 0.0
        ):
            raise ValueError("Geçersiz zaman veya varyans.")


class Gyroscope:

    def __init__(
        self,
        config: GyroConfig | None = None,
        seed: int = 101,
    ):
        self.config = (
            config if config is not None else GyroConfig()
        )
        self._rng = random.Random(seed)
        self.time_s = 0.0

    def measure(
        self,
        true_yaw_rate: float,
        dt: float,
    ) -> GyroMeasurement:
        """
        Bu aralıktaki ortalama gerçek açısal hızı alır.

        Gerçek hız ve bias, üretilen pakete eklenmez.
        """
        if not (
            math.isfinite(true_yaw_rate)
            and math.isfinite(dt)
            and dt > 0.0
        ):
            raise ValueError("Geçersiz açısal hız veya süre.")

        next_time = self.time_s + dt

        if not math.isfinite(next_time):
            raise ValueError("Ölçüm zamanı sonlu olmalıdır.")

        noise = self._rng.gauss(
            0.0, self.config.noise_std_rad_s
        )

        packet = GyroMeasurement(
            time_s=next_time,
            dt_s=dt,
            yaw_rate_rad_s=(
                true_yaw_rate
                + self.config.bias_rad_s
                + noise
            ),
            nominal_variance=(
                self.config.noise_std_rad_s ** 2
            ),
        )

        self.time_s = next_time
        return packet
    