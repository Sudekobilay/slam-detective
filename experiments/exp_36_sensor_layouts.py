"""EXP-36: JSON'dan LiDAR düzeni ve ortak EKF-SLAM arayüzü."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import groupby
import json
import math
from pathlib import Path

from estimation.slam_backend import LineSlamBackend
from experiments.exp_24_motion_comparison import (
    write_json,
    write_jsonl,
)
from experiments.exp_30_scan_guard_audit import (
    fixed_recording,
)
from experiments.line_slam_common import (
    evaluate,
    write_csv,
)
from simulator.encoder import EncoderMeasurement
from simulator.environment import create_demo_environment
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.lidar import LidarConfig
from simulator.measurement_stream import measurement_to_event
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit
from simulator.robot import Pose


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


def make_recording(data, seed=801):
    units = read_layout(data)

    base, records = fixed_recording(
        "arc",
        seed,
        0.0,
    )

    events = [
        event
        for event in base
        if (
            event["sensor_type"] != "lidar"
            and (
                data["gps_enabled"]
                or event["sensor_type"] != "gps"
            )
        )
    ]

    truth = {
        round(row["time_s"], 12): row
        for row in records["truth"]
    }

    times = sorted({
        event["time_s"]
        for event in base
        if event["sensor_type"] == "lidar"
    })

    environment = create_demo_environment()

    for time in times:
        row = truth[time]

        pose = Pose(
            row["x_m"],
            row["y_m"],
            row["theta_rad"],
        )

        for unit in units:
            event = measurement_to_event(
                unit.measure(
                    pose,
                    environment,
                    time,
                )
            )

            events.append(json.loads(json.dumps(
                event,
                allow_nan=False,
            )))

    events.sort(
        key=lambda event: (
            event["time_s"],
            event["sensor_type"],
            event["sensor_id"],
        )
    )

    return events, records


def replay_layout(
    events,
    data,
    backend_factory=LineSlamBackend,
):
    mounts = {
        unit.sensor_id: unit.actual_mount
        for unit in read_layout(data)
    }

    backend = backend_factory(mounts)
    path = []
    last = -1.0

    for time, batch in groupby(
        events,
        key=lambda event: event["time_s"],
    ):
        if time <= last:
            raise ValueError(
                "Zaman sırası geçersiz."
            )

        batch = list(batch)

        keys = [
            (
                event["sensor_type"],
                event["sensor_id"],
            )
            for event in batch
        ]

        if len(set(keys)) != len(keys):
            raise ValueError(
                "Tekrarlanan sensör paketi."
            )

        encoder = next(
            (
                event
                for event in batch
                if event["sensor_type"] == "encoder"
            ),
            None,
        )

        gyro = next(
            (
                event
                for event in batch
                if event["sensor_type"] == "gyro"
            ),
            None,
        )

        if (encoder is None) != (gyro is None):
            raise ValueError(
                "Encoder/gyro çifti gerekli."
            )

        if encoder is not None:
            backend.predict(
                EncoderMeasurement(
                    time_s=time,
                    **encoder["data"],
                ),
                GyroMeasurement(
                    time_s=time,
                    **gyro["data"],
                ),
            )

        for event in batch:
            if event["sensor_type"] == "gps":
                if not data["gps_enabled"]:
                    raise ValueError(
                        "Kapalı GPS için ölçüm bulundu."
                    )

                backend.update_gps(
                    GPSMeasurement(
                        time_s=time,
                        **event["data"],
                    )
                )

            elif event["sensor_type"] == "lidar":
                backend.update_scan(
                    event["sensor_id"],
                    event["data"]["scan"]["points"],
                )

        path.append(
            backend.snapshot(time)
        )
        last = time

    return backend, path


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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if args.config:
        paths = [args.config]

    else:
        folder = root / "config" / "sensors"
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = []

        for name, data in examples().items():
            path = folder / f"{name}.json"

            if not path.exists():
                write_json(path, data)

            paths.append(path)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = root / "results" / f"exp_36_{stamp}"
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "EXP-36: yapılandırılabilir LiDAR düzeni; "
        "temel EKF-SLAM, teşhis kapalı.",
        flush=True,
    )

    summaries = []
    reference_truth = None

    for index, path in enumerate(paths):
        data = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )

        units = read_layout(data)

        events, records = make_recording(data)
        backend, estimate = replay_layout(
            events,
            data,
        )

        folder = output / f"layout_{index + 1}"
        folder.mkdir()

        write_json(
            folder / "used_config.json",
            data,
        )
        write_jsonl(
            folder / "measurements.jsonl",
            events,
        )

        truth = folder / "ground_truth.csv"

        write_csv(
            truth,
            records["truth"],
        )

        _, evaluated, metrics = evaluate(
            {"line_slam": estimate},
            truth,
        )

        write_csv(
            folder / "evaluation.csv",
            evaluated["line_slam"],
        )

        write_json(
            folder / "map.json",
            [
                asdict(feature)
                for feature
                in backend.estimator.map_features()
            ],
        )

        expected = {
            "encoder": 40,
            "gyro": 40,
            **(
                {"gps": 21}
                if data["gps_enabled"]
                else {}
            ),
            **{
                unit.sensor_id: 9
                for unit in units
            },
        }

        checks = {
            "packet_counts": (
                dict(Counter(
                    event["sensor_id"]
                    for event in events
                )) == expected
            ),
            "unique_source_time": (
                len({
                    (
                        event["sensor_type"],
                        event["sensor_id"],
                        event["time_s"],
                    )
                    for event in events
                }) == len(events)
            ),
            "same_true_motion": (
                reference_truth is None
                or reference_truth == records["truth"]
            ),
        }

        reference_truth = records["truth"]

        if not all(checks.values()):
            raise RuntimeError(
                "Sensör düzeni kontrolleri başarısız."
            )

        row = {
            "name": data["name"],
            "lidar_count": len(units),
            "gps_enabled": data["gps_enabled"],
            "rays_per_scan_instant": sum(
                unit.config.beam_count
                for unit in units
            ),
            "packet_count": len(events),
            "map_line_count": (
                backend.estimator.landmark_count
            ),
            "position_rmse_m": metrics[
                "line_slam"
            ]["position_rmse_m"],
            "all_checks_passed": all(
                checks.values()
            ),
        }

        summaries.append(row)

        write_json(
            folder / "summary.json",
            {
                **row,
                "checks": checks,
            },
        )

        print(
            f"{data['name']}: "
            f"LiDAR={len(units)} | "
            f"GPS={data['gps_enabled']} | "
            f"paket={len(events)} | "
            f"çizgi={row['map_line_count']} | "
            f"RMSE={row['position_rmse_m']:.6f} m | PASS",
            flush=True,
        )

    write_csv(
        output / "group_summary.csv",
        summaries,
    )

    write_json(
        output / "summary.json",
        {
            "experiment_id": "exp_36",
            "layouts": summaries,
            "limitations": [
                "Temel haritalama ve yapılandırma "
                "kontrolüdür; çoklu sensör teşhisi "
                "veya hareket seçimi deneyi değildir.",
                "Mevcut teşhis modülleri front/rear "
                "düzenine özeldir; keyfi LiDAR "
                "sayılarına henüz bağlanmadı.",
                "Encoder ve gyro bu arayüzde "
                "zorunludur; GPS açılıp kapatılabilir.",
                "GPS gürültüsü .05 m ve kesinti "
                "[1,3) sabittir; bu dosyalar yalnız "
                "yerleşimi ve GPS kullanımını "
                "yapılandırır.",
                "Örnek düzenlerin ışın sayısı "
                "tam eşit değildir; en iyi "
                "yerleşim veya adil bütçeli "
                "başarı sıralaması çıkarılmaz.",
                "Tek ortam, ideal LiDAR, hatasız "
                "montaj ve nominal başlangıç; "
                "burada GPS veya LiDAR koruması "
                "uygulanmaz.",
                "Örnek JSON dosyaları yalnız "
                "yoksa oluşturulur; mevcut kullanıcı "
                "ayarlarının üzerine yazılmaz.",
            ],
        },
    )

    print("Sensör düzeni kontrolleri: PASS")
    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()