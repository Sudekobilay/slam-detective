"""Sensör ölçümlerini ortak, gerçek bilgiden ayrı kayıt biçimine taşır."""

from dataclasses import asdict
import math

from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.multi_lidar import LidarMeasurement


def measurement_to_event(packet) -> dict:
    """
    Yalnız tanımlı ölçüm sınıflarını kabul eder.

    Gerçek poz, gerçek montaj ve hata etiketleri alınmaz.
    """
    packet_types = {
        EncoderMeasurement: ("encoder", "encoder"),
        GyroMeasurement: ("gyro", "gyro"),
        GPSMeasurement: ("gps", "gps"),
        LidarMeasurement: ("lidar", None),
    }

    if type(packet) not in packet_types:
        raise TypeError("Desteklenmeyen ölçüm paketi.")

    if not math.isfinite(packet.time_s) or packet.time_s < 0.0:
        raise ValueError("Geçersiz ölçüm zamanı.")

    sensor_type, sensor_id = packet_types[type(packet)]

    data = asdict(packet)
    data.pop("time_s")

    if sensor_type == "lidar":
        sensor_id = data.pop("sensor_id")

    # Ortak simülasyon saatindeki kayan nokta artıklarını
    # normalize eder. Gerçek sensör gecikmesi modeli değildir.
    time_s = round(packet.time_s, 12)

    return {
        "schema_version": 1,
        "time_s": time_s,
        "sensor_type": sensor_type,
        "sensor_id": sensor_id,
        "data": data,
    }