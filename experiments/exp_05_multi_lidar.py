"""İki LiDAR'da tek sensör montaj hatasının kontrollü incelenmesi."""

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.exp_04_mounting_error import draw_environment
from simulator.environment import create_demo_environment
from simulator.lidar import LidarConfig
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit, scan_all
from simulator.robot import Pose


def main():
    environment = create_demo_environment()
    robot_pose = Pose(1.0, 3.0, 0.0)

    config = LidarConfig(
        field_of_view=math.pi,
        beam_count=91,
        max_range=15.0,
    )

    # Yazılımın bildiği montajlar: iki durumda da değişmiyor.
    assumed_mounts = {
        "front": SensorMount(x=0.25, yaw=0.0),
        "rear": SensorMount(x=-0.25, yaw=math.pi),
    }

    cases = (
        ("healthy", 0.0, "İki LiDAR sağlam"),
        ("front_yaw_error", 5.0, "Ön LiDAR'da 5° montaj hatası"),
    )

    project_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_directory = (
        project_root / "results" / f"exp_05_{run_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    figure, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    summary = {
        "experiment": "EXP-05: Çoklu LiDAR ve tek montaj hatası",
        "run_id": run_id,
        "robot_pose": asdict(robot_pose),
        "lidar_config": asdict(config),
        "assumed_mounts": {
            sensor_id: asdict(mount)
            for sensor_id, mount in assumed_mounts.items()
        },
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [
                asdict(obstacle)
                for obstacle in environment.obstacles
            ],
        },
        "cases": {},
        "scope": (
            "İdeal, eşzamanlı, sabit pozlu geometri deneyi. "
            "Gerçek montaj ve robot pozu yalnız simülasyon "
            "üretimi ile çevrimdışı değerlendirmede kullanılır. "
            "Otomatik teşhis veya SLAM değildir."
        ),
    }

    checks = {}
    rear_reference = None

    for axis, (case_name, front_error_deg, title) in zip(
        axes, cases
    ):
        units = (
            LidarUnit(
                sensor_id="front",
                actual_mount=SensorMount(
                    x=0.25,
                    yaw=math.radians(front_error_deg),
                ),
                config=config,
            ),
            LidarUnit(
                sensor_id="rear",
                actual_mount=SensorMount(
                    x=-0.25,
                    yaw=math.pi,
                ),
                config=config,
            ),
        )

        packets = scan_all(
            units,
            robot_pose,
            environment,
            time_s=0.0,
        )

        actual_mounts = {
            unit.sensor_id: unit.actual_mount
            for unit in units
        }

        # Arka sensörün ham paketi iki durumda aynı olmalı.
        rear_packet = next(
            packet
            for packet in packets
            if packet.sensor_id == "rear"
        )

        if rear_reference is None:
            rear_reference = rear_packet
        else:
            checks["rear_raw_measurements_unchanged"] = (
                rear_packet == rear_reference
            )

        draw_environment(axis, environment)
        rows = []
        case_metrics = {}

        for packet in packets:
            sensor_id = packet.sensor_id
            actual_mount = actual_mounts[sensor_id]
            assumed_mount = assumed_mounts[sensor_id]

            true_points = []
            reconstructed_points = []
            errors = []
            analytical_differences = []

            angle_difference = (
                actual_mount.yaw - assumed_mount.yaw
            )

            for index, (angle, distance, point) in enumerate(
                zip(
                    packet.scan.angles,
                    packet.scan.ranges,
                    packet.scan.points,
                    strict=True,
                )
            ):
                true_point = None
                reconstructed = None
                error = None
                expected_error = None

                if point is not None:
                    # Gerçek dönüşüm: yalnız değerlendirme.
                    true_point = actual_mount.point_to_world(
                        point, robot_pose
                    )

                    # Aynı ölçüm, varsayılan montajla yorumlanır.
                    reconstructed = assumed_mount.point_to_world(
                        point, robot_pose
                    )

                    error = math.dist(true_point, reconstructed)

                    # Bu deneyde yalnız yaw farklı;
                    # gerçek ve varsayılan sensör merkezleri aynı.
                    expected_error = (
                        2.0
                        * distance
                        * abs(math.sin(angle_difference / 2.0))
                    )

                    errors.append(error)
                    analytical_differences.append(
                        abs(error - expected_error)
                    )
                    true_points.append(true_point)
                    reconstructed_points.append(reconstructed)

                rows.append({
                    "sensor_id": sensor_id,
                    "time_s": packet.time_s,
                    "beam_index": index,
                    "angle_sensor_rad": angle,
                    "hit": point is not None,
                    "range_m": distance,
                    "true_world_x_m": (
                        true_point[0]
                        if true_point is not None else None
                    ),
                    "true_world_y_m": (
                        true_point[1]
                        if true_point is not None else None
                    ),
                    "reconstructed_world_x_m": (
                        reconstructed[0]
                        if reconstructed is not None else None
                    ),
                    "reconstructed_world_y_m": (
                        reconstructed[1]
                        if reconstructed is not None else None
                    ),
                    "point_error_m": error,
                    "analytical_error_m": expected_error,
                })

            if not errors:
                raise RuntimeError(
                    f"{case_name}/{sensor_id}: geçerli ölçüm yok."
                )

            rmse = math.sqrt(
                sum(error**2 for error in errors) / len(errors)
            )

            case_metrics[sensor_id] = {
                "beam_count": len(packet.scan.ranges),
                "valid_return_count": len(errors),
                "point_rmse_m": rmse,
                "maximum_point_error_m": max(errors),
            }

            checks[f"{case_name}_{sensor_id}_formula"] = (
                max(analytical_differences) < 1e-9
            )

            should_be_correct = (
                sensor_id == "rear" or front_error_deg == 0.0
            )
            checks[f"{case_name}_{sensor_id}_effect"] = (
                rmse < 1e-9 if should_be_correct else rmse > 1e-3
            )

            color = "royalblue" if sensor_id == "front" else "darkorange"
            label = "Ön LiDAR" if sensor_id == "front" else "Arka LiDAR"

            axis.scatter(
                [point[0] for point in true_points],
                [point[1] for point in true_points],
                s=25,
                facecolors="none",
                edgecolors=color,
                alpha=0.6,
                label=f"{label}: gerçek",
                zorder=3,
            )

            axis.scatter(
                [point[0] for point in reconstructed_points],
                [point[1] for point in reconstructed_points],
                s=16,
                marker="x",
                color=color,
                label=f"{label}: yorumlanan",
                zorder=4,
            )

            print(
                f"{case_name}/{sensor_id}: "
                f"nokta RMSE={rmse:.6f} m"
            )

        with (output_directory / f"{case_name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        summary["cases"][case_name] = {
            "actual_mounts": {
                sensor_id: asdict(mount)
                for sensor_id, mount in actual_mounts.items()
            },
            "metrics": case_metrics,
        }

        axis.scatter(
            robot_pose.x,
            robot_pose.y,
            color="black",
            s=40,
            label="Robot merkezi",
            zorder=5,
        )
        axis.set(
            title=title,
            xlabel="x (m)",
            ylabel="y (m)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="upper right")

    # Görsel karşılaştırma için aynı sınırlar.
    x_limits = (
        min(axis.get_xlim()[0] for axis in axes),
        max(axis.get_xlim()[1] for axis in axes),
    )
    y_limits = (
        min(axis.get_ylim()[0] for axis in axes),
        max(axis.get_ylim()[1] for axis in axes),
    )

    for axis in axes:
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)

    figure.suptitle(
        "SLAM-Detective — Sensör Kimliği Korunan Çoklu LiDAR",
        fontsize=14,
    )
    figure.savefig(
        output_directory / "multi_lidar.png",
        dpi=160,
    )
    plt.close(figure)

    summary["checks"] = checks
    summary["all_passed"] = all(checks.values())

    with (output_directory / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(
        "Deney kontrolleri: "
        + ("PASS" if summary["all_passed"] else "FAIL")
    )
    print(f"Sonuç klasörü: {output_directory}")

    if not summary["all_passed"]:
        raise RuntimeError(
            "Çoklu LiDAR deneyindeki kontroller başarısız."
        )


if __name__ == "__main__":
    main()