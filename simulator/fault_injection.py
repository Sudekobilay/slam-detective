"""Kaydedilmiş ölçümlerin kopyasına kontrollü GPS sapması ekler."""

from copy import deepcopy
import math


def inject_gps_bias(
    events,
    start_s,
    end_s,
    bias_x_m,
    bias_y_m,
):
    """
    Girdi kayıtlarını değiştirmez.

    Dönen ölçüm akışına gerçek hata etiketi eklemez.
    Aralık: [start_s, end_s).
    """
    values = (start_s, end_s, bias_x_m, bias_y_m)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Hata parametreleri sonlu olmalıdır.")

    if not 0.0 <= start_s < end_s:
        raise ValueError("Geçersiz hata zaman aralığı.")

    modified = deepcopy(events)
    affected_count = 0

    for event in modified:
        if event["sensor_type"] != "gps":
            continue

        time_s = event["time_s"]

        if not math.isfinite(time_s):
            raise ValueError("Geçersiz GPS zamanı.")

        if (
            start_s <= time_s < end_s
            and event["data"]["available"]
        ):
            event["data"]["x_m"] += bias_x_m
            event["data"]["y_m"] += bias_y_m
            affected_count += 1

    return modified, affected_count