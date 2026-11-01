"""Düz ve dönüşlü hareketlerde zaman damgalı çoklu LiDAR kayıtları."""

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
from simulator.collision import try_step
from simulator.environment import create_demo_environment
from simulator.lidar import LidarConfig
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit, scan_all
from simulator.robot import DifferentialDriveRobot, Pose


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    environment = create_demo_environment()
    initial_pose = Pose(1.0, 3.0, 0.0)

    dt = 0.1
    steps = 40
    body_radius = 0.35

    config = LidarConfig(
        field_of_view=math.pi,
        beam_count=91,
        max_range=15.0,
    )

    assumed_mounts = {
        "front": SensorMount(x=0.25),
        "rear": SensorMount(x=-0.25, yaw=math.pi),
    }

    # İki komutta da v=0.2 m/s.
    motions = {
        "straight": {
            "rates": (4.0, 4.0),
            "expected": Pose(1.8, 3.0, 0.0),
            "title": "Düz hareket",
        },
        "arc": {
            "rates": (3.1, 4.9),
            "expected": Pose(
                1.0 + (0.2 / 0.3) * math.sin(1.2),
                3.0 + (0.2 / 0.3) * (1.0 - math.cos(1.2)),
                1.2,
            ),
            "title": "Yay dönüşü",
        },
    }

    project_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_directory = (
        project_root / "results" / f"exp_06_{run_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    figure, axes = plt.subplots(
        2, 2,
        figsize=(14, 11),
        constrained_layout=True,
    )

    summary = {
        "experiment": "EXP-06: Hareketli çoklu LiDAR",
        "run_id": run_id,
        "initial_pose": asdict(initial_pose),
        "dt_s": dt,
        "steps": steps,
        "duration_s": steps * dt,
        "commanded_center_path_length_m": 0.8,
        "body_radius_m": body_radius,
        "lidar_config": asdict(config),
        "robot_geometry": asdict(
            DifferentialDriveRobot().geometry
        ),
        "assumed_mounts": {
            key: asdict(value)
            for key, value in assumed_mounts.items()
        },
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [
                asdict(obstacle)
                for obstacle in environment.obstacles
            ],
        },
        "cases": {},
        "checks": {},
        "limitations": [
            "İdeal ve eşzamanlı ölçümler.",
            "Her tarama anlık kabul edilir; hareket bozulması yok.",
            "Gerçek poz yalnız üretim ve değerlendirme içindir.",
            "SLAM, otomatik teşhis ve hareket seçimi henüz yok.",
            "Grafikte yalnız son tarama noktaları gösterilir.",
        ],
    }

    for row_index, (motion_name, motion) in enumerate(
        motions.items()
    ):
        healthy_trajectory = None
        healthy_rear_packets = None

        for column_index, yaw_deg in enumerate((0.0, 5.0)):
            case_name = (
                f"{motion_name}_"
                + ("healthy" if yaw_deg == 0.0 else "front_fault")
            )
            case_directory = output_directory / case_name
            case_directory.mkdir()

            units = (
                LidarUnit(
                    sensor_id="front",
                    actual_mount=SensorMount(
                        x=0.25,
                        yaw=math.radians(yaw_deg),
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
            actual_mounts = {
                unit.sensor_id: unit.actual_mount
                for unit in units
            }

            robot = DifferentialDriveRobot(
                initial_pose=initial_pose
            )
            trajectory = []
            metrics = []
            rear_packets = []
            final_points = {}

            left_rate, right_rate = motion["rates"]

            with (case_directory / "measurements.jsonl").open(
                "w", encoding="utf-8"
            ) as measurement_file:

                # t=0 ölçümü dahil toplam 41 tarama anı.
                for index in range(steps + 1):
                    if index > 0:
                        accepted = try_step(
                            robot,
                            environment,
                            left_rate,
                            right_rate,
                            dt,
                            radius=body_radius,
                        )

                        if not accepted:
                            raise RuntimeError(
                                f"{case_name}: hareket reddedildi; "
                                f"adım={index}"
                            )

                    time_s = index * dt
                    pose = robot.pose

                    trajectory.append({
                        "time_s": time_s,
                        "x_m": pose.x,
                        "y_m": pose.y,
                        "theta_rad": pose.theta,
                    })

                    packets = scan_all(
                        units, pose, environment, time_s
                    )

                    for packet in packets:
                        # Bu akış gerçek poz veya montaj içermez.
                        measurement_file.write(
                            json.dumps(
                                asdict(packet),
                                ensure_ascii=False,
                                allow_nan=False,
                            ) + "\n"
                        )

                        if packet.sensor_id == "rear":
                            rear_packets.append(packet)

                        sid = packet.sensor_id
                        errors = []
                        truth_points = []
                        interpreted_points = []

                        for point in packet.scan.points:
                            if point is None:
                                continue

                            truth_point = (
                                actual_mounts[sid].point_to_world(
                                    point, pose
                                )
                            )
                            interpreted = (
                                assumed_mounts[sid].point_to_world(
                                    point, pose
                                )
                            )

                            errors.append(
                                math.dist(truth_point, interpreted)
                            )
                            truth_points.append(truth_point)
                            interpreted_points.append(interpreted)

                        metrics.append({
                            "time_s": time_s,
                            "sensor_id": sid,
                            "valid_return_count": len(errors),
                            "point_rmse_m": (
                                math.sqrt(
                                    sum(e * e for e in errors)
                                    / len(errors)
                                )
                                if errors else None
                            ),
                        })

                        final_points[sid] = (
                            truth_points,
                            interpreted_points,
                        )

            write_csv(
                case_directory / "ground_truth.csv",
                trajectory,
            )
            write_csv(
                case_directory / "evaluation_metrics.csv",
                metrics,
            )

            expected = motion["expected"]
            position_error = math.hypot(
                robot.pose.x - expected.x,
                robot.pose.y - expected.y,
            )
            heading_difference = (
                robot.pose.theta - expected.theta
            )
            heading_error = abs(math.atan2(
                math.sin(heading_difference),
                math.cos(heading_difference),
            ))

            checks = {
                "analytical_endpoint": (
                    position_error < 1e-9
                    and heading_error < 1e-9
                ),
                "sample_count": len(trajectory) == 41,
                "rear_scan_count": len(rear_packets) == 41,
                "all_scans_have_returns": all(
                    record["valid_return_count"] > 0
                    for record in metrics
                ),
            }

            for sid in ("front", "rear"):
                values = [
                    record["point_rmse_m"]
                    for record in metrics
                    if record["sensor_id"] == sid
                    and record["point_rmse_m"] is not None
                ]

                should_be_correct = (
                    sid == "rear" or yaw_deg == 0.0
                )
                checks[f"{sid}_expected_effect"] = (
                    bool(values)
                    and (
                        max(values) < 1e-9
                        if should_be_correct
                        else min(values) > 1e-3
                    )
                )

            if yaw_deg == 0.0:
                healthy_trajectory = trajectory
                healthy_rear_packets = rear_packets
            else:
                checks["same_physical_trajectory"] = (
                    trajectory == healthy_trajectory
                )
                checks["rear_measurements_unchanged"] = (
                    rear_packets == healthy_rear_packets
                )

            summary["cases"][case_name] = {
                "wheel_rates_rad_s": {
                    "left": left_rate,
                    "right": right_rate,
                },
                "actual_mounts": {
                    key: asdict(value)
                    for key, value in actual_mounts.items()
                },
                "expected_final_pose": asdict(expected),
                "actual_final_pose": asdict(robot.pose),
                "endpoint_position_error_m": position_error,
                "endpoint_heading_error_rad": heading_error,
                "measurement_packet_count": len(metrics),
                "checks": checks,
            }

            summary["checks"][case_name] = all(
                checks.values()
            )

            axis = axes[row_index, column_index]
            draw_environment(axis, environment)

            axis.plot(
                [sample["x_m"] for sample in trajectory],
                [sample["y_m"] for sample in trajectory],
                color="black",
                linewidth=2,
                label="Gerçek rota",
            )
            axis.scatter(
                initial_pose.x,
                initial_pose.y,
                color="green",
                marker="o",
                s=35,
                label="Başlangıç",
                zorder=5,
            )

            for sid, color in (
                ("front", "royalblue"),
                ("rear", "darkorange"),
            ):
                truth_points, interpreted = final_points[sid]

                axis.scatter(
                    [p[0] for p in truth_points],
                    [p[1] for p in truth_points],
                    s=15,
                    facecolors="none",
                    edgecolors=color,
                    alpha=0.5,
                )
                axis.scatter(
                    [p[0] for p in interpreted],
                    [p[1] for p in interpreted],
                    s=12,
                    marker="x",
                    color=color,
                    label=f"{sid}: son tarama",
                )

            state = "Sağlam" if yaw_deg == 0.0 else "Ön LiDAR +5°"
            axis.set(
                title=f"{motion['title']} — {state}",
                xlabel="x (m)",
                ylabel="y (m)",
                xlim=(-1, 11),
                ylim=(-1, 9),
            )
            axis.set_aspect("equal")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7, loc="upper right")

            status = "PASS" if all(checks.values()) else "FAIL"
            print(
                f"{case_name}: {status} | "
                f"tarama anı={len(trajectory)} | "
                f"ölçüm paketi={len(metrics)}"
            )

    summary["all_passed"] = all(
        summary["checks"].values()
    )

    figure.suptitle(
        "SLAM-Detective — Hareketli LiDAR Kayıtları",
        fontsize=15,
    )
    figure.savefig(
        output_directory / "moving_lidar.png",
        dpi=160,
    )
    plt.close(figure)

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

    print(f"Sonuç klasörü: {output_directory}")

    if not summary["all_passed"]:
        raise RuntimeError(
            "Hareketli LiDAR deneyinin kontrolleri başarısız."
        )


if __name__ == "__main__":
    main()