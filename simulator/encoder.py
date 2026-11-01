"""Tekerlek dönüşlerinden tam sayı encoder ölçümleri üretir."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EncoderConfig:
    # Bir tam tekerlek dönüşündeki etkin sayım adedi.
    # Ham PPR değil; varsa quadrature etkisi dahil kabul edilir.
    counts_per_revolution: int = 2048

    def __post_init__(self):
        if (
            type(self.counts_per_revolution) is not int
            or self.counts_per_revolution <= 0
        ):
            raise ValueError(
                "Bir turdaki sayım adedi pozitif tam sayı olmalıdır."
            )


@dataclass(frozen=True)
class EncoderMeasurement:
    """Bir zaman aralığında ölçülen darbe artışları."""

    time_s: float
    dt_s: float
    left_delta_counts: int
    right_delta_counts: int

    def __post_init__(self):
        if not (
            math.isfinite(self.time_s)
            and math.isfinite(self.dt_s)
            and self.dt_s > 0.0
            and self.time_s >= self.dt_s
        ):
            raise ValueError("Geçersiz encoder zaman bilgisi.")

        if (
            type(self.left_delta_counts) is not int
            or type(self.right_delta_counts) is not int
        ):
            raise ValueError(
                "Encoder sayımları tam sayı olmalıdır."
            )


class WheelEncoder:
    """Başlangıçta sıfırlanmış, ideal çift tekerlek encoder'ı."""

    def __init__(self, config: EncoderConfig | None = None):
        self.config = (
            config if config is not None else EncoderConfig()
        )
        self.time_s = 0.0
        self._left_angle = 0.0
        self._right_angle = 0.0
        self._left_count = 0
        self._right_count = 0

    def measure(
        self,
        left_rate: float,
        right_rate: float,
        dt: float,
    ) -> EncoderMeasurement:
        """
        Gerçek tekerlek mili hızlarını alır: rad/s.

        Robotun gerçek konumunu veya zemindeki ilerlemesini almaz.
        """
        if not all(
            math.isfinite(value)
            for value in (left_rate, right_rate, dt)
        ) or dt <= 0.0:
            raise ValueError(
                "Hızlar sonlu, süre pozitif ve sonlu olmalıdır."
            )

        left_angle = self._left_angle + left_rate * dt
        right_angle = self._right_angle + right_rate * dt
        next_time = self.time_s + dt

        if not all(
            math.isfinite(value)
            for value in (left_angle, right_angle, next_time)
        ):
            raise ValueError("Encoder birikimi sonlu olmalıdır.")

        scale = self.config.counts_per_revolution / (2.0 * math.pi)

        # Her adımdaki küçük dönüşü ayrı ayrı yuvarlamıyoruz.
        # Birikmiş mil açısını sayıya çevirerek kesirleri koruyoruz.
        left_count = round(left_angle * scale)
        right_count = round(right_angle * scale)

        measurement = EncoderMeasurement(
            time_s=next_time,
            dt_s=dt,
            left_delta_counts=left_count - self._left_count,
            right_delta_counts=right_count - self._right_count,
        )

        self._left_angle = left_angle
        self._right_angle = right_angle
        self._left_count = left_count
        self._right_count = right_count
        self.time_s = next_time

        return measurement