"""Gyro gürültüsü ve bias'ın yön tahminine etkisi."""

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from estimation.gyro_heading import GyroHeading
from simulator.collision import try_step
from simulator.environment import create_demo_environment
from simulator.imu import GyroConfig, Gyroscope
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
    seed = 101

    motions = {
        "straight": (4.0, 4.0, "Düz hareket"),
        "arc": (3.1, 4.9, "Yay dönüşü"),
    }

    configurations = {
        "ideal": GyroConfig(),
        "noise": GyroConfig(noise_std_rad_s=0.01),
        "noise_bias": GyroConfig(
            noise_std_rad_s=0.01,
            bias_rad_s=0.03,
        ),
    }

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_08_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    figure, axes = plt.subplots(
        2, 2, figsize=(13, 9), constrained_layout=True
    )

    summary = {
        "experiment": "EXP-08: Gyro yön tahmini",
        "run_id": run_id,
        "seed": seed,
        "dt_s": dt,
        "steps": steps,
        "initial_pose": asdict(initial_pose),
        "body_radius_m": 0.35,
        "robot_geometry": asdict(
            DifferentialDriveRobot().geometry
        ),
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [
                asdict(item) for item in environment.obstacles
            ],
        },
        "cases": {},
        "checks": {},
        "limitations": [
            "Yalnız düzlemsel gyro; ivmeölçer yok.",
            "Bilinen başlangıç yönü.",
            "Sabit örnek aralığı; gürültü std örnek başına.",
            "Tek seed; istatistiksel üstünlük iddiası yok.",
            "Bias kestirimi veya sensör füzyonu yok.",
        ],
    }

    for column, (motion_name, motion) in enumerate(motions.items()):
        left, right, title = motion
        reference_truth = None
        noise_heading = None

        for case_name, config in configurations.items():
            robot = DifferentialDriveRobot(initial_pose=initial_pose)
            gyro = Gyroscope(config, seed=seed)
            heading = GyroHeading(initial_pose.theta)

            geometry = robot.geometry
            true_rate = (
                geometry.right_radius * right
                - geometry.left_radius * left
            ) / geometry.wheelbase

            truth = [{
                "time_s": 0.0,
                "x_m": robot.pose.x,
                "y_m": robot.pose.y,
                "theta_rad": robot.pose.theta,
            }]
            evaluation = [{
                "time_s": 0.0,
                "estimated_theta_rad": heading.theta,
                "heading_error_rad": 0.0,
            }]
            packets = []

            for index in range(1, steps + 1):
                if not try_step(
                    robot, environment, left, right, dt,
                    radius=0.35,
                ):
                    raise RuntimeError("Hareket reddedildi.")

                # Sabit tekerlek hızları altında gerçek açısal hız.
                packet = gyro.measure(true_rate, dt)
                heading.update(packet)
                packets.append(asdict(packet))

                difference = heading.theta - robot.pose.theta
                error = math.atan2(
                    math.sin(difference), math.cos(difference)
                )

                truth.append({
                    "time_s": index * dt,
                    "x_m": robot.pose.x,
                    "y_m": robot.pose.y,
                    "theta_rad": robot.pose.theta,
                })
                evaluation.append({
                    "time_s": index * dt,
                    "estimated_theta_rad": heading.theta,
                    "heading_error_rad": error,
                })

            checks = {
                "packet_count": len(packets) == steps,
                "finite_estimates": all(
                    math.isfinite(value)
                    for row in evaluation
                    for value in row.values()
                ),
                "timestamps_aligned": all(
                    math.isclose(
                        packet["time_s"], (index + 1) * dt,
                        rel_tol=0.0, abs_tol=1e-9,
                    )
                    for index, packet in enumerate(packets)
                ),
            }

            if reference_truth is None:
                reference_truth = truth
            else:
                checks["same_ground_truth"] = (
                    truth == reference_truth
                )

            if case_name == "ideal":
                checks["ideal_heading_matches"] = all(
                    abs(row["heading_error_rad"]) < 1e-9
                    for row in evaluation
                )

            if case_name == "noise":
                noise_heading = [
                    row["estimated_theta_rad"]
                    for row in evaluation
                ]

            if case_name == "noise_bias":
                # Aynı seed ile gürültü örnekleri eşleştirilmiştir.
                # Bias'lı ve bias'sız yön farkı b × t olmalı.
                checks["bias_effect_matches"] = all(
                    math.isclose(
                        row["estimated_theta_rad"] - base,
                        config.bias_rad_s * row["time_s"],
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for row, base in zip(
                        evaluation, noise_heading, strict=True
                    )
                )

            rmse = math.sqrt(
                sum(row["heading_error_rad"] ** 2
                    for row in evaluation) / len(evaluation)
            )

            case_id = f"{motion_name}_{case_name}"
            folder = output / case_id
            folder.mkdir()

            write_csv(folder / "gyro.csv", packets)
            write_csv(folder / "ground_truth.csv", truth)
            write_csv(folder / "heading_evaluation.csv", evaluation)

            summary["cases"][case_id] = {
                "config": asdict(config),
                "wheel_rates_rad_s": {"left": left, "right": right},
                "heading_rmse_rad": rmse,
                "final_heading_error_rad": (
                    evaluation[-1]["heading_error_rad"]
                ),
                "checks": checks,
            }
            summary["checks"][case_id] = all(checks.values())

            times = [row["time_s"] for row in evaluation]

            axes[0, column].plot(
                times,
                [math.degrees(row["estimated_theta_rad"])
                 for row in evaluation],
                label=case_name,
            )
            axes[1, column].plot(
                times,
                [math.degrees(row["heading_error_rad"])
                 for row in evaluation],
                label=case_name,
            )

            status = "PASS" if all(checks.values()) else "FAIL"
            print(
                f"{case_id}: {status} | "
                f"yön RMSE={rmse:.6f} rad | "
                f"son yön hatası="
                f"{evaluation[-1]['heading_error_rad']:.6f} rad"
            )

        axes[0, column].plot(
            [row["time_s"] for row in reference_truth],
            [math.degrees(row["theta_rad"])
             for row in reference_truth],
            "k--", label="Gerçek yön",
        )
        axes[0, column].set(
            title=title, xlabel="Zaman (s)", ylabel="Yön (derece)"
        )
        axes[1, column].set(
            title=f"{title} — Yön hatası",
            xlabel="Zaman (s)", ylabel="Yön hatası (derece)",
        )

        for row_index in (0, 1):
            axes[row_index, column].grid(alpha=0.25)
            axes[row_index, column].legend(fontsize=8)

    summary["all_passed"] = all(summary["checks"].values())

    figure.suptitle("SLAM-Detective — Gyro Yön Tahmini")
    figure.savefig(output / "gyro_comparison.png", dpi=160)
    plt.close(figure)

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    print(f"Sonuç klasörü: {output}")

    if not summary["all_passed"]:
        raise RuntimeError("Gyro deney kontrolleri başarısız.")


if __name__ == "__main__":
    main()