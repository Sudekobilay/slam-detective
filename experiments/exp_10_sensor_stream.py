"""Encoder, gyro, GPS ve iki LiDAR'ı ortak saatle kaydeder."""

import argparse
from collections import Counter
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from simulator.collision import try_step
from simulator.encoder import EncoderConfig, WheelEncoder
from simulator.environment import create_demo_environment
from simulator.gps import GPSConfig, GPSSensor
from simulator.imu import GyroConfig, Gyroscope
from simulator.lidar import LidarConfig
from simulator.measurement_stream import measurement_to_event
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit, scan_all
from simulator.robot import DifferentialDriveRobot, Pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--front-yaw-error-deg",
        type=float,
        default=0.0,
        help="Ön LiDAR'ın gerçek montajına eklenecek açı.",
    )
    parser.add_argument(
        "--fault-start-s",
        type=float,
        default=2.0,
        help="Montaj değişiminin başlangıç zamanı.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Eşleştirilmiş sağlıklı EXP-10 sonuç klasörü.",
    )
    args = parser.parse_args()

    if (
        not math.isfinite(args.front_yaw_error_deg)
        or abs(args.front_yaw_error_deg) >= 180
    ):
        parser.error("Montaj hatası sonlu ve mutlak değeri 180° altında olmalı.")

    if (
        not math.isfinite(args.fault_start_s)
        or not 0.0 <= args.fault_start_s <= 4.0
    ):
        parser.error("Hata başlangıcı 0–4 saniye arasında olmalı.")

    environment = create_demo_environment()
    initial_pose = Pose(1.0, 3.0, 0.0)
    robot = DifferentialDriveRobot(initial_pose=initial_pose)

    dt = 0.1
    steps = 40
    left_rate, right_rate = 3.1, 4.9

    encoder_config = EncoderConfig()
    gyro_config = GyroConfig(noise_std_rad_s=0.01)
    gps_config = GPSConfig(
        noise_std_m=0.05,
        outage_start_s=1.0,
        outage_end_s=3.0,
    )
    lidar_config = LidarConfig(
        field_of_view=math.pi,
        beam_count=91,
        max_range=15.0,
    )

    encoder = WheelEncoder(encoder_config)
    gyro = Gyroscope(gyro_config, seed=101)
    gps = GPSSensor(gps_config, seed=102)

    nominal_units = (
        LidarUnit(
            "front",
            SensorMount(x=0.25),
            lidar_config,
        ),
        LidarUnit(
            "rear",
            SensorMount(x=-0.25, yaw=math.pi),
            lidar_config,
        ),
    )

    faulty_units = (
        LidarUnit(
            "front",
            SensorMount(
                x=0.25,
                yaw=math.radians(args.front_yaw_error_deg),
            ),
            lidar_config,
        ),
        nominal_units[1],
    )

    fault_enabled = args.front_yaw_error_deg != 0.0

    geometry = robot.geometry
    true_yaw_rate = (
        geometry.right_radius * right_rate
        - geometry.left_radius * left_rate
    ) / geometry.wheelbase

    events = []
    truth_rows = []
    affected_scan_times = []

    for index in range(steps + 1):
        if index > 0:
            if not try_step(
                robot, environment,
                left_rate, right_rate, dt,
                radius=0.35,
            ):
                raise RuntimeError("Hareket reddedildi.")

            events.append(measurement_to_event(
                encoder.measure(left_rate, right_rate, dt)
            ))
            events.append(measurement_to_event(
                gyro.measure(true_yaw_rate, dt)
            ))

        time_s = index * dt

        truth_rows.append({
            "time_s": time_s,
            "x_m": robot.pose.x,
            "y_m": robot.pose.y,
            "theta_rad": robot.pose.theta,
        })

        if index % 2 == 0:
            events.append(measurement_to_event(
                gps.measure(robot.pose, time_s)
            ))

        if index % 5 == 0:
            fault_active = (
                fault_enabled
                and time_s >= args.fault_start_s
            )
            actual_units = (
                faulty_units if fault_active else nominal_units
            )

            if fault_active:
                affected_scan_times.append(time_s)

            for packet in scan_all(
                actual_units, robot.pose, environment, time_s
            ):
                # Paketlere hata etiketi veya gerçek montaj eklenmez.
                events.append(measurement_to_event(packet))

    events.sort(key=lambda event: (
        event["time_s"],
        event["sensor_type"],
        event["sensor_id"],
    ))

        # Yeni kayıtları, JSON'dan okunan referansla aynı veri
    # türlerine dönüştürür. Sayısal değerleri değiştirmez.
    events = json.loads(json.dumps(events, allow_nan=False))

    counts = Counter(event["sensor_id"] for event in events)
    expected_counts = {
        "encoder": 40,
        "gyro": 40,
        "gps": 21,
        "front": 9,
        "rear": 9,
    }

    gps_events = [
        event for event in events
        if event["sensor_type"] == "gps"
    ]
    missing_gps = [
        event for event in gps_events
        if not event["data"]["available"]
    ]

    checks = {
        "packet_counts": dict(counts) == expected_counts,
        "total_packets": len(events) == 119,
        "missing_gps_count": len(missing_gps) == 10,
        "ordered_timestamps": all(
            first["time_s"] <= second["time_s"]
            for first, second in zip(events, events[1:])
        ),
        "outage_matches": all(
            event["data"]["available"]
            == (not (1.0 <= event["time_s"] < 3.0))
            for event in gps_events
        ),
        "no_duplicate_source_time": len({
            (
                event["sensor_type"],
                event["sensor_id"],
                event["time_s"],
            )
            for event in events
        }) == len(events),
    }

    # Referans yalnız deney kontrolünde kullanılır.
    # Bu bölüm kestirim veya teşhis yapmaz.
    if args.reference is not None:
        reference_measurements = (
            args.reference / "measurements.jsonl"
        )
        reference_truth = args.reference / "ground_truth.csv"

        reference_events = [
            json.loads(line)
            for line in reference_measurements.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        with reference_truth.open(
            newline="", encoding="utf-8"
        ) as file:
            reference_truth_rows = [
                {
                    key: float(value)
                    for key, value in row.items()
                }
                for row in csv.DictReader(file)
            ]

        def event_key(event):
            return (
                event["sensor_type"],
                event["sensor_id"],
                event["time_s"],
            )

        reference_by_key = {
            event_key(event): event
            for event in reference_events
        }
        current_by_key = {
            event_key(event): event
            for event in events
        }

        checks["reference_packet_keys_match"] = (
            reference_by_key.keys() == current_by_key.keys()
        )

        checks["ground_truth_unchanged"] = (
            truth_rows == reference_truth_rows
        )

        checks["other_sensors_unchanged"] = (
            [
                event for event in events
                if event["sensor_id"] != "front"
            ]
            == [
                event for event in reference_events
                if event["sensor_id"] != "front"
            ]
        )

        checks["front_before_fault_unchanged"] = all(
            reference_by_key.get(event_key(event)) == event
            for event in events
            if (
                event["sensor_id"] == "front"
                and event["time_s"] < args.fault_start_s
            )
        )

        if fault_enabled:
            checks["front_after_fault_changed"] = any(
                reference_by_key.get(event_key(event)) != event
                for event in events
                if (
                    event["sensor_id"] == "front"
                    and event["time_s"] >= args.fault_start_s
                )
            )
        else:
            checks["all_measurements_unchanged"] = (
                events == reference_events
            )

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_10_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    with (output / "measurements.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for event in events:
            file.write(json.dumps(
                event,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n")

    with (output / "ground_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=truth_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(truth_rows)

    summary = {
        "experiment": "EXP-10: Ortak sensör akışı",
        "run_id": run_id,
        "initial_pose": asdict(initial_pose),
        "robot_geometry": asdict(geometry),
        "body_radius_m": 0.35,
        "motion_dt_s": dt,
        "steps": steps,
        "wheel_rates_rad_s": {
            "left": left_rate,
            "right": right_rate,
        },
        "rates_hz": {
            "encoder": 10,
            "gyro": 10,
            "gps": 5,
            "lidar_each": 2,
        },
        "encoder_config": asdict(encoder_config),
        "gyro_config": asdict(gyro_config),
        "gps_config": asdict(gps_config),
        "gyro_seed": 101,
        "gps_seed": 102,
        # Başlangıçtaki gerçek montajlar.
        "lidar_units": [
            asdict(unit) for unit in nominal_units
        ],
        "mounting_change": {
            "enabled": fault_enabled,
            "sensor_id": "front",
            "requested_start_s": args.fault_start_s,
            "yaw_error_deg": args.front_yaw_error_deg,
            "affected_scan_times_s": affected_scan_times,
            "actual_mount_after_change": asdict(
                faulty_units[0].actual_mount
            ),
            "estimator_assumed_mount": asdict(
                nominal_units[0].actual_mount
            ),
        },
        "reference_directory": (
            str(args.reference)
            if args.reference is not None else None
        ),
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [
                asdict(item) for item in environment.obstacles
            ],
        },
        "counts": dict(counts),
        "checks": checks,
        "all_passed": all(checks.values()),
        "limitations": [
            "Bu dosya yalnız ölçüm üretir; SLAM veya teşhis uygulamaz.",
            "Ortak saat, sıfır gecikme ve anlık LiDAR taraması.",
            "GPS kesinti paketleri konum ölçümü değildir.",
            "Montaj değişimi ani ve yalnız açısaldır.",
            "Hata, başlangıçtan sonraki ilk uygun taramada görünür.",
            "summary ve ground_truth yalnız değerlendirme içindir.",
            "Ölçüm zamanları 12 ondalık basamağa normalize edilir.",
        ],
    }

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    figure, axis = plt.subplots(
        figsize=(12, 5), constrained_layout=True
    )
    sensor_ids = ["encoder", "gyro", "gps", "front", "rear"]

    for lane, sensor_id in enumerate(sensor_ids):
        selected = [
            event for event in events
            if event["sensor_id"] == sensor_id
        ]
        valid = [
            event for event in selected
            if event["data"].get("available", True)
        ]
        missing = [
            event for event in selected
            if not event["data"].get("available", True)
        ]

        axis.scatter(
            [event["time_s"] for event in valid],
            [lane] * len(valid),
            s=22,
            color="royalblue",
        )
        axis.scatter(
            [event["time_s"] for event in missing],
            [lane] * len(missing),
            s=40,
            marker="x",
            color="crimson",
        )

    axis.scatter([], [], color="royalblue", label="Ölçüm mevcut")
    axis.scatter(
        [], [], color="crimson",
        marker="x", label="GPS ölçümü yok",
    )

    if fault_enabled:
        axis.axvline(
            args.fault_start_s,
            color="darkorange",
            linestyle="--",
            label="Ön LiDAR montaj değişimi",
        )

    axis.set_yticks(range(len(sensor_ids)), sensor_ids)
    axis.set(
        title="SLAM-Detective — Ortak Sensör Zaman Çizelgesi",
        xlabel="Zaman (s)",
        ylabel="Sensör",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output / "sensor_timeline.png", dpi=160)
    plt.close(figure)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"Toplam kayıt: {len(events)}")
    print(f"Eksik GPS ölçümü: {len(missing_gps)}")
    print(f"Ön LiDAR açı hatası: {args.front_yaw_error_deg:.2f} derece")
    print(f"Etkilenen ön LiDAR taraması: {len(affected_scan_times)}")
    print(f"Sonuç klasörü: {output}")

    if not summary["all_passed"]:
        raise RuntimeError("Ortak kayıt kontrolleri başarısız.")


if __name__ == "__main__":
    main()