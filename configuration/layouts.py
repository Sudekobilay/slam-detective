"""Validated LiDAR layouts, independent of experiment runners."""
import math
from simulator.lidar import LidarConfig
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit

def read_layout(data):
    if (
        set(data) - {"name", "gps_enabled", "lidars"}
        or not isinstance(data.get("name"), str)
        or not data["name"].strip()
    ):
        raise ValueError(
            "Geçersiz düzen adı veya alan."
        )

    if (
        type(data.get("gps_enabled")) is not bool
        or not isinstance(data.get("lidars"), list)
        or not data["lidars"]
    ):
        raise ValueError(
            "GPS seçimi bool, LiDAR listesi "
            "boş olmayan liste olmalı."
        )

    units = []
    seen = set()

    allowed = {
        "id",
        "x_m",
        "y_m",
        "yaw_deg",
        "fov_deg",
        "beams",
        "max_range_m",
    }

    for item in data["lidars"]:
        if set(item) != allowed:
            raise ValueError(
                "LiDAR alanları eksik veya tanımsız."
            )

        name = item["id"]

        if (
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or name in seen
            or name in ("encoder", "gyro", "gps")
        ):
            raise ValueError(
                "LiDAR kimliği geçersiz veya tekrarlı."
            )

        numeric_fields = (
            "x_m",
            "y_m",
            "yaw_deg",
            "fov_deg",
            "max_range_m",
        )

        if any(
            type(item[key]) not in (int, float)
            or not math.isfinite(item[key])
            for key in numeric_fields
        ):
            raise ValueError(
                "LiDAR geometrisi sonlu sayı olmalı."
            )

        seen.add(name)

        units.append(
            LidarUnit(
                name,
                SensorMount(
                    item["x_m"],
                    item["y_m"],
                    math.radians(item["yaw_deg"]),
                ),
                LidarConfig(
                    math.radians(item["fov_deg"]),
                    item["beams"],
                    item["max_range_m"],
                ),
            )
        )

    return tuple(units)


def examples():
    def sensor(name, x, y, yaw, fov, beams):
        return {
            "id": name,
            "x_m": x,
            "y_m": y,
            "yaw_deg": yaw,
            "fov_deg": fov,
            "beams": beams,
            "max_range_m": 15.0,
        }

    return {
        "single": {
            "name": "single",
            "gps_enabled": True,
            "lidars": [
                sensor(
                    "center",
                    0.0, 0.0, 0.0,
                    360.0, 181,
                ),
            ],
        },
        "dual": {
            "name": "dual",
            "gps_enabled": True,
            "lidars": [
                sensor(
                    "front",
                    0.25, 0.0, 0.0,
                    180.0, 91,
                ),
                sensor(
                    "rear",
                    -0.25, 0.0, 180.0,
                    180.0, 91,
                ),
            ],
        },
        "quad": {
            "name": "quad",
            "gps_enabled": True,
            "lidars": [
                sensor(
                    "front",
                    0.25, 0.0, 0.0,
                    90.0, 46,
                ),
                sensor(
                    "rear",
                    -0.25, 0.0, 180.0,
                    90.0, 46,
                ),
                sensor(
                    "left",
                    0.0, 0.25, 90.0,
                    90.0, 46,
                ),
                sensor(
                    "right",
                    0.0, -0.25, -90.0,
                    90.0, 46,
                ),
            ],
        },
    }
