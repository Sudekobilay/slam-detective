"""Gerçek rota ile encoder odometrisinin karşılaştırılması."""

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from estimation.odometry import WheelOdometry
from simulator.collision import try_step
from simulator.encoder import EncoderConfig, WheelEncoder
from simulator.environment import create_demo_environment
from simulator.robot import (
    DifferentialDriveRobot,
    Pose,
    RobotGeometry,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    environment = create_demo_environment()
    initial_pose = Pose(1.0, 3.0, 0.0)
    true_geometry = RobotGeometry()

    dt = 0.1
    steps = 40

    motions = {
        "straight": (4.0, 4.0, "Düz hareket"),
        "arc": (3.1, 4.9, "Yay dönüşü"),
    }

    cases = {
        "fine_counts": (
            2048,
            RobotGeometry(),
            "2048 sayım/tur",
        ),
        "coarse_counts": (
            100,
            RobotGeometry(),
            "100 sayım/tur",
        ),
        "left_radius_error": (
            2048,
            RobotGeometry(left_radius=0.0525),
            "Sol yarıçap kalibrasyon hatası",
        ),
    }

    project_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_directory = (
        project_root / "results" / f"exp_07_{run_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    figure, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        constrained_layout=True,
    )

    summary = {
        "experiment": "EXP-07: Encoder ve odometri",
        "run_id": run_id,
        "initial_pose": asdict(initial_pose),
        "true_geometry": asdict(true_geometry),
        "dt_s": dt,
        "steps": steps,
        "body_radius_m": 0.35,
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
            "Bilinen ortak başlangıç pozu.",
            "Gerçek mil hızları komut hızlarına eşit.",
            "Encoder gürültüsü, kayma ve paket kaybı yok.",
            "2048 sayım/tur durumu da kuantalama içerir.",
            "Tek deterministik rota gerçekleşimi; "
            "genel performans sıralaması değildir.",
        ],
    }

    for column, (motion_name, motion) in enumerate(
        motions.items()
    ):
        left_rate, right_rate, title = motion
        reference_truth = None

        for case_name, case in cases.items():
            counts_per_revolution, assumed_geometry, label = case

            encoder_config = EncoderConfig(
                counts_per_revolution=counts_per_revolution
            )
            encoder = WheelEncoder(encoder_config)

            # Fiziksel robot daima aynı gerçek geometriyi kullanır.
            robot = DifferentialDriveRobot(
                geometry=true_geometry,
                initial_pose=initial_pose,
            )

            # Tahmin yalnız encoder paketlerinden güncellenir.
            odometry = WheelOdometry(
                geometry=assumed_geometry,
                encoder_config=encoder_config,
                initial_pose=initial_pose,
            )

            case_id = f"{motion_name}_{case_name}"
            case_directory = output_directory / case_id
            case_directory.mkdir()

            truth_rows = []
            estimate_rows = []
            packets = []

            for index in range(steps + 1):
                if index > 0:
                    accepted = try_step(
                        robot,
                        environment,
                        left_rate,
                        right_rate,
                        dt,
                        radius=0.35,
                    )

                    if not accepted:
                        raise RuntimeError(
                            f"{case_id}: hareket reddedildi."
                        )

                    # Bu deneyde ideal mil hızları komuta eşit.
                    packet = encoder.measure(
                        left_rate, right_rate, dt
                    )
                    odometry.update(packet)
                    packets.append(asdict(packet))

                time_s = index * dt

                position_error = math.hypot(
                    odometry.pose.x - robot.pose.x,
                    odometry.pose.y - robot.pose.y,
                )

                angle_difference = (
                    odometry.pose.theta - robot.pose.theta
                )
                heading_error = math.atan2(
                    math.sin(angle_difference),
                    math.cos(angle_difference),
                )

                truth_rows.append({
                    "time_s": time_s,
                    "x_m": robot.pose.x,
                    "y_m": robot.pose.y,
                    "theta_rad": robot.pose.theta,
                })

                estimate_rows.append({
                    "time_s": time_s,
                    "x_m": odometry.pose.x,
                    "y_m": odometry.pose.y,
                    "theta_rad": odometry.pose.theta,
                    "position_error_m": position_error,
                    "heading_error_rad": heading_error,
                })

            checks = {
                "sample_count": len(estimate_rows) == 41,
                "encoder_packet_count": len(packets) == 40,
                "timestamps_aligned": all(
                    math.isclose(
                        packet["time_s"],
                        (index + 1) * dt,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for index, packet in enumerate(packets)
                ),
                "finite_estimates": all(
                    math.isfinite(value)
                    for row in estimate_rows
                    for value in row.values()
                ),
            }

            if reference_truth is None:
                reference_truth = truth_rows
            else:
                checks["same_ground_truth"] = (
                    truth_rows == reference_truth
                )

            position_rmse = math.sqrt(
                sum(
                    row["position_error_m"] ** 2
                    for row in estimate_rows
                ) / len(estimate_rows)
            )
            heading_rmse = math.sqrt(
                sum(
                    row["heading_error_rad"] ** 2
                    for row in estimate_rows
                ) / len(estimate_rows)
            )

            summary["cases"][case_id] = {
                "wheel_rates_rad_s": {
                    "left": left_rate,
                    "right": right_rate,
                },
                "counts_per_revolution": counts_per_revolution,
                "assumed_geometry": asdict(assumed_geometry),
                "position_rmse_m": position_rmse,
                "heading_rmse_rad": heading_rmse,
                "final_position_error_m": (
                    estimate_rows[-1]["position_error_m"]
                ),
                "checks": checks,
            }

            summary["checks"][case_id] = all(checks.values())

            write_csv(
                case_directory / "ground_truth.csv",
                truth_rows,
            )
            write_csv(
                case_directory / "encoder.csv",
                packets,
            )
            write_csv(
                case_directory / "odometry_evaluation.csv",
                estimate_rows,
            )

            axes[0, column].plot(
                [row["x_m"] for row in estimate_rows],
                [row["y_m"] for row in estimate_rows],
                label=label,
            )
            axes[1, column].plot(
                [row["time_s"] for row in estimate_rows],
                [row["position_error_m"] for row in estimate_rows],
                label=label,
            )

            status = "PASS" if all(checks.values()) else "FAIL"

            print(
                f"{case_id}: {status} | "
                f"konum RMSE={position_rmse:.6f} m | "
                f"yön RMSE={heading_rmse:.6f} rad"
            )

        axes[0, column].plot(
            [row["x_m"] for row in reference_truth],
            [row["y_m"] for row in reference_truth],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Gerçek rota",
        )

        axes[0, column].set(
            title=title,
            xlabel="x (m)",
            ylabel="y (m)",
        )
        axes[0, column].set_aspect(
            "equal", adjustable="datalim"
        )

        axes[1, column].set(
            title=f"{title} — Konum hatası",
            xlabel="Zaman (s)",
            ylabel="Konum hatası (m)",
        )

        for row_index in (0, 1):
            axes[row_index, column].grid(alpha=0.25)
            axes[row_index, column].legend(fontsize=8)

    summary["all_passed"] = all(
        summary["checks"].values()
    )

    figure.suptitle(
        "SLAM-Detective — Gerçek Rota ve Encoder Odometrisi",
        fontsize=15,
    )
    figure.savefig(
        output_directory / "odometry_comparison.png",
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
            "Odometri deneyindeki kontroller başarısız."
        )


if __name__ == "__main__":
    main()