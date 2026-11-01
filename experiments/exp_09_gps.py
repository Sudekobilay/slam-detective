"""Aynı fiziksel rotada dört GPS koşulunun karşılaştırılması."""

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
from simulator.environment import create_demo_environment
from simulator.gps import GPSConfig, GPSSensor
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

    configurations = {
        "ideal": GPSConfig(),
        "noise": GPSConfig(noise_std_m=0.05),
        "noise_bias": GPSConfig(
            noise_std_m=0.05,
            bias_x_m=0.5,
            bias_y_m=-0.3,
        ),
        "outage": GPSConfig(
            noise_std_m=0.05,
            outage_start_s=1.0,
            outage_end_s=3.0,
        ),
    }

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_09_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 6), constrained_layout=True
    )

    summary = {
        "experiment": "EXP-09: GPS ölçümleri",
        "run_id": run_id,
        "seed": seed,
        "initial_pose": asdict(initial_pose),
        "robot_geometry": asdict(
            DifferentialDriveRobot().geometry
        ),
        "motion_dt_s": dt,
        "steps": steps,
        "wheel_rates_rad_s": {"left": 3.1, "right": 4.9},
        "gps_rate_hz": 5.0,
        "body_radius_m": 0.35,
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [
                asdict(item) for item in environment.obstacles
            ],
        },
        "cases": {},
        "checks": {},
        "limitations": [
            "Yerel 2B konum modeli; uydu fiziği modellenmiyor.",
            "Sensör robot merkezinde.",
            "GPS ölçümleri gösteriliyor; filtreleme yapılmıyor.",
            "RMSE yalnız geçerli GPS ölçümlerinde hesaplanır.",
            "Tek seed; genel istatistiksel sonuç değildir.",
        ],
    }

    reference_truth = None

    for case_name, config in configurations.items():
        robot = DifferentialDriveRobot(initial_pose=initial_pose)
        sensor = GPSSensor(config, seed=seed)
        truth = []
        measurements = []
        evaluation = []

        for index in range(steps + 1):
            if index > 0:
                if not try_step(
                    robot, environment, 3.1, 4.9, dt,
                    radius=0.35,
                ):
                    raise RuntimeError("Hareket reddedildi.")

            time_s = index * dt

            truth.append({
                "time_s": time_s,
                "x_m": robot.pose.x,
                "y_m": robot.pose.y,
                "theta_rad": robot.pose.theta,
            })

            # Hareket 10 Hz, GPS örneklemesi 5 Hz.
            if index % 2 != 0:
                continue

            packet = sensor.measure(robot.pose, time_s)
            measurements.append(asdict(packet))

            error = (
                math.hypot(
                    packet.x_m - robot.pose.x,
                    packet.y_m - robot.pose.y,
                )
                if packet.available else None
            )

            evaluation.append({
                "time_s": time_s,
                "available": packet.available,
                "position_error_m": error,
            })

        errors = [
            row["position_error_m"]
            for row in evaluation
            if row["available"]
        ]
        valid_count = len(errors)

        checks = {
            "scheduled_count": len(measurements) == 21,
            "valid_count": valid_count == (
                11 if case_name == "outage" else 21
            ),
        }

        if reference_truth is None:
            reference_truth = truth
        else:
            checks["same_ground_truth"] = (
                truth == reference_truth
            )

        if case_name == "ideal":
            checks["ideal_positions_match"] = max(errors) < 1e-9

        if case_name == "outage":
            checks["outage_interval_matches"] = all(
                row["available"]
                == (not (1.0 <= row["time_s"] < 3.0))
                for row in measurements
            )

        rmse = math.sqrt(
            sum(error**2 for error in errors) / valid_count
        )

        folder = output / case_name
        folder.mkdir()
        write_csv(folder / "gps.csv", measurements)
        write_csv(folder / "ground_truth.csv", truth)
        write_csv(folder / "gps_evaluation.csv", evaluation)

        summary["cases"][case_name] = {
            "config": asdict(config),
            "scheduled_count": len(measurements),
            "valid_count": valid_count,
            "missing_count": len(measurements) - valid_count,
            "valid_measurement_rmse_m": rmse,
            "checks": checks,
        }
        summary["checks"][case_name] = all(checks.values())

        valid_packets = [
            row for row in measurements if row["available"]
        ]

        axes[0].scatter(
            [row["x_m"] for row in valid_packets],
            [row["y_m"] for row in valid_packets],
            s=22,
            label=case_name,
        )
        axes[1].plot(
            [row["time_s"] for row in evaluation],
            [row["position_error_m"] for row in evaluation],
            marker=".",
            label=case_name,
        )

        status = "PASS" if all(checks.values()) else "FAIL"
        print(
            f"{case_name}: {status} | "
            f"geçerli={valid_count}/21 | "
            f"ölçüm RMSE={rmse:.6f} m"
        )

    axes[0].plot(
        [row["x_m"] for row in reference_truth],
        [row["y_m"] for row in reference_truth],
        "k--", label="Gerçek rota",
    )
    axes[0].set(
        title="GPS konum ölçümleri",
        xlabel="x (m)", ylabel="y (m)",
    )
    axes[0].set_aspect("equal", adjustable="datalim")

    axes[1].axvspan(
        1.0, 3.0, color="gray", alpha=0.15,
        label="outage koşulunun kesinti aralığı",
    )
    axes[1].set(
        title="Geçerli GPS ölçümlerinin hatası",
        xlabel="Zaman (s)", ylabel="Konum hatası (m)",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle("SLAM-Detective — GPS Modeli")
    figure.savefig(output / "gps_comparison.png", dpi=160)
    plt.close(figure)

    summary["all_passed"] = all(summary["checks"].values())

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    print(f"Sonuç klasörü: {output}")

    if not summary["all_passed"]:
        raise RuntimeError("GPS deney kontrolleri başarısız.")


if __name__ == "__main__":
    main()