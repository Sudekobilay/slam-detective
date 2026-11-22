"""Shared sensor configurations and validated measurement values."""
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
