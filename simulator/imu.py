"""Düzlemsel IMU'nun gyro bileşeni."""

from domain.measurements import GyroConfig, GyroMeasurement
import math
import random






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
    