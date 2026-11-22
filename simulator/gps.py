"""Yerel 2B koordinatlarda GPS konum ölçümü."""

from domain.measurements import GPSConfig, GPSMeasurement
import math
import random

from simulator.robot import Pose






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