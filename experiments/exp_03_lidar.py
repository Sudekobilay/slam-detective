"""İdeal LiDAR taramasını çizer; ölçümleri CSV ve JSON'a kaydeder."""

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as PlotRectangle

from simulator.environment import create_demo_environment
from simulator.lidar import Lidar, LidarConfig
from simulator.robot import Pose


def sensor_point_to_world(point, sensor_pose):
    """Yalnız görselleştirme için sensör noktasını dünyaya taşır."""
    cosine = math.cos(sensor_pose.theta)
    sine = math.sin(sensor_pose.theta)

    return (
        sensor_pose.x + cosine * point[0] - sine * point[1],
        sensor_pose.y + sine * point[0] + cosine * point[1],
    )


def main():
    environment = create_demo_environment()

    # Bu aşamada LiDAR robot merkezinde, robotla aynı yönde.
    sensor_pose = Pose(1.0, 3.0, 0.0)

    config = LidarConfig(
        field_of_view=2.0 * math.pi,
        beam_count=180,
        max_range=15.0,
    )

    sensor = Lidar(config)
    scan = sensor.scan(sensor_pose, environment)

    project_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_directory = (
        project_root / "results" / f"exp_03_{run_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    rows = []
    world_points = []

    for index, (angle, distance, point) in enumerate(
        zip(scan.angles, scan.ranges, scan.points)
    ):
        world_point = (
            sensor_point_to_world(point, sensor_pose)
            if point is not None
            else None
        )

        if world_point is not None:
            world_points.append(world_point)

        rows.append({
            "beam_index": index,
            "angle_sensor_rad": angle,
            "angle_sensor_deg": math.degrees(angle),
            "hit": distance is not None,
            "range_m": distance,
            "point_sensor_x_m": (
                point[0] if point is not None else None
            ),
            "point_sensor_y_m": (
                point[1] if point is not None else None
            ),
            "point_world_x_m": (
                world_point[0]
                if world_point is not None else None
            ),
            "point_world_y_m": (
                world_point[1]
                if world_point is not None else None
            ),
        })

    with (output_directory / "scan.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    valid_ranges = [
        distance
        for distance in scan.ranges
        if distance is not None
    ]

    # 180 ışınlık tam taramada 90. ışın +x yönündedir.
    front_index = min(
        range(len(scan.angles)),
        key=lambda index: abs(scan.angles[index]),
    )
    front_range = scan.ranges[front_index]

    checks = {
        "beam_count": len(scan.ranges) == 180,
        "all_beams_hit": len(valid_ranges) == 180,
        "front_angle_zero": math.isclose(
            scan.angles[front_index],
            0.0,
            abs_tol=1e-12,
        ),
        "front_obstacle_at_2m": (
            front_range is not None
            and math.isclose(
                front_range,
                2.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ),
    }

    figure, axis = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    for obstacle in environment.obstacles:
        axis.add_patch(
            PlotRectangle(
                (obstacle.x_min, obstacle.y_min),
                obstacle.x_max - obstacle.x_min,
                obstacle.y_max - obstacle.y_min,
                facecolor="lightgray",
                edgecolor="none",
            )
        )

    for wall in environment.walls:
        axis.plot(
            [wall.start[0], wall.end[0]],
            [wall.start[1], wall.end[1]],
            color="black",
            linewidth=2,
            zorder=3,
        )

    for index, point in enumerate(world_points):
        axis.plot(
            [sensor_pose.x, point[0]],
            [sensor_pose.y, point[1]],
            color="dodgerblue",
            alpha=0.22,
            linewidth=0.7,
            label="LiDAR ışınları" if index == 0 else None,
            zorder=1,
        )

    if world_points:
        axis.scatter(
            [point[0] for point in world_points],
            [point[1] for point in world_points],
            color="crimson",
            s=10,
            label="Ölçülen yüzey noktaları",
            zorder=4,
        )

    axis.scatter(
        sensor_pose.x,
        sensor_pose.y,
        color="royalblue",
        s=85,
        label="LiDAR merkezi",
        zorder=5,
    )

    axis.arrow(
        sensor_pose.x,
        sensor_pose.y,
        0.6 * math.cos(sensor_pose.theta),
        0.6 * math.sin(sensor_pose.theta),
        head_width=0.15,
        length_includes_head=True,
        color="darkorange",
        zorder=6,
    )

    axis.set(
        title="SLAM-Detective — İdeal LiDAR Taraması",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(-0.5, 10.5),
        ylim=(-0.5, 8.5),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")

    figure.savefig(
        output_directory / "lidar_scan.png",
        dpi=160,
    )
    plt.close(figure)

    record = {
        "experiment": "EXP-03: İdeal LiDAR taraması",
        "run_id": run_id,
        "sensor_pose": asdict(sensor_pose),
        "config": asdict(config),
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [
                asdict(obstacle)
                for obstacle in environment.obstacles
            ],
        },
        "metrics": {
            "beam_count": len(scan.ranges),
            "valid_return_count": len(valid_ranges),
            "no_return_count": (
                len(scan.ranges) - len(valid_ranges)
            ),
            "minimum_range_m": (
                min(valid_ranges) if valid_ranges else None
            ),
            "maximum_range_m": (
                max(valid_ranges) if valid_ranges else None
            ),
            "mean_range_m": (
                sum(valid_ranges) / len(valid_ranges)
                if valid_ranges else None
            ),
            "front_range_m": front_range,
        },
        "checks": checks,
        "all_passed": all(checks.values()),
        "coordinate_note": (
            "scan.points sensör koordinatlarındadır. "
            "Dünya noktaları yalnız görselleştirme ve "
            "simülasyon doğrulaması için hesaplanmıştır."
        ),
        "limitations": [
            "İdeal ve gürültüsüz ölçümler.",
            "Sabit robot; tarama sırasında hareket yok.",
            "LiDAR robot merkezinde kabul ediliyor.",
            "SLAM ve harita tahmini uygulanmıyor.",
        ],
    }

    with (output_directory / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"Işın sayısı: {len(scan.ranges)}")
    print(f"Geçerli ölçüm: {len(valid_ranges)}")
    print(f"Ön yöndeki mesafe: {front_range} m")
    print(f"Sonuç klasörü: {output_directory}")

    if not record["all_passed"]:
        raise RuntimeError(
            "LiDAR deneyindeki kontroller başarısız."
        )


if __name__ == "__main__":
    main()