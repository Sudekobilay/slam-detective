"""İdeal robot hareketlerini çizer ve deney verilerini kaydeder."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from simulator.robot import DifferentialDriveRobot


def build_scenarios():
    """Komutlar: sol hız (rad/s), sağ hız (rad/s), süre (s)."""
    return {
        "straight": {
            "title": "Düz hareket",
            "commands": [(4.0, 4.0, 5.0)],
            "expected": (1.0, 0.0, 0.0),
        },
        "reverse": {
            "title": "Geri hareket",
            "commands": [(-4.0, -4.0, 5.0)],
            "expected": (-1.0, 0.0, 0.0),
        },
        "spin": {
            "title": "Yerinde dönüş",
            "commands": [(-3.0, 3.0, math.pi / 2)],
            "expected": (0.0, 0.0, math.pi / 2),
        },
        "quarter_circle": {
            "title": "Çeyrek çember",
            "commands": [(17.0, 23.0, math.pi / 2)],
            "expected": (1.0, 1.0, math.pi / 2),
        },
        "full_circle": {
            "title": "Tam çember",
            "commands": [(17.0, 23.0, 2 * math.pi)],
            "expected": (0.0, 0.0, 0.0),
        },
        "square": {
            "title": "Kare rota",
            "commands": [
                (4.0, 4.0, 5.0),
                (-3.0, 3.0, math.pi / 2),
            ] * 4,
            "expected": (0.0, 0.0, 0.0),
        },
    }


def simulate(commands, max_dt=0.05):
    robot = DifferentialDriveRobot()
    elapsed = 0.0

    rows = [{
        "time_s": elapsed,
        "x_m": robot.pose.x,
        "y_m": robot.pose.y,
        "theta_rad": robot.pose.theta,
    }]

    for left_rate, right_rate, duration in commands:
        # Süreyi yuvarlamadan eşit alt adımlara böl.
        number_of_steps = max(1, math.ceil(duration / max_dt))
        dt = duration / number_of_steps

        for _ in range(number_of_steps):
            pose = robot.step(left_rate, right_rate, dt)
            elapsed += dt

            rows.append({
                "time_s": elapsed,
                "x_m": pose.x,
                "y_m": pose.y,
                "theta_rad": pose.theta,
            })

    return rows


def main():
    project_root = Path(__file__).resolve().parents[1]

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_directory = (
        project_root / "results" / f"exp_01_{run_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    scenarios = build_scenarios()
    max_dt = 0.05
    tolerance = 1e-9

    summary = {
        "experiment": "EXP-01: İdeal robot hareketi",
        "run_id": run_id,
        "units": {
            "position": "m",
            "angle": "rad",
            "time": "s",
            "wheel_rate": "rad/s",
        },
        "geometry": {
            "left_radius_m": 0.05,
            "right_radius_m": 0.05,
            "wheelbase_m": 0.30,
        },
        "max_dt_s": max_dt,
        "position_tolerance_m": tolerance,
        "heading_tolerance_rad": tolerance,
        "randomness": "Yok; ideal deterministik model",
        "scenarios": {},
    }

    figure, axes = plt.subplots(
        2, 3,
        figsize=(12, 8),
        constrained_layout=True,
    )

    for axis, (name, scenario) in zip(
        axes.flat, scenarios.items()
    ):
        rows = simulate(scenario["commands"], max_dt)

        csv_path = output_directory / f"{name}.csv"

        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        final = rows[-1]
        expected_x, expected_y, expected_theta = (
            scenario["expected"]
        )

        position_error = math.hypot(
            final["x_m"] - expected_x,
            final["y_m"] - expected_y,
        )

        angle_difference = (
            final["theta_rad"] - expected_theta
        )
        heading_error = abs(math.atan2(
            math.sin(angle_difference),
            math.cos(angle_difference),
        ))

        passed = (
            position_error <= tolerance
            and heading_error <= tolerance
        )

        summary["scenarios"][name] = {
            "commands": [
                {
                    "left_rad_s": left,
                    "right_rad_s": right,
                    "duration_s": duration,
                }
                for left, right, duration
                in scenario["commands"]
            ],
            "expected_pose": {
                "x_m": expected_x,
                "y_m": expected_y,
                "theta_rad": expected_theta,
            },
            "final_sample": final,
            "position_error_m": position_error,
            "heading_error_rad": heading_error,
            "passed": passed,
        }

        axis.plot(
            [row["x_m"] for row in rows],
            [row["y_m"] for row in rows],
            color="royalblue",
            linewidth=2,
            label="Rota",
        )
        axis.scatter(
            0, 0,
            color="green",
            s=65,
            label="Başlangıç",
            zorder=3,
        )
        axis.scatter(
            final["x_m"], final["y_m"],
            color="crimson",
            marker="x",
            s=80,
            label="Bitiş",
            zorder=4,
        )

        if name == "spin":
            # Robot yer değiştirmediği için yönleri göster.
            for index in (0, len(rows) // 2, len(rows) - 1):
                angle = rows[index]["theta_rad"]
                axis.arrow(
                    0, 0,
                    0.3 * math.cos(angle),
                    0.3 * math.sin(angle),
                    head_width=0.035,
                    length_includes_head=True,
                    alpha=0.7,
                )

        axis.set_title(scenario["title"])
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

        status = "PASS" if passed else "FAIL"
        print(
            f"{name}: {status} | "
            f"konum hatası={position_error:.3e} m | "
            f"yön hatası={heading_error:.3e} rad"
        )

    summary["all_passed"] = all(
        result["passed"]
        for result in summary["scenarios"].values()
    )

    figure.suptitle(
        "SLAM-Detective — İdeal Robot Hareketi",
        fontsize=16,
    )
    figure.savefig(
        output_directory / "robot_motion.png",
        dpi=160,
    )
    plt.close(figure)

    with (output_directory / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"\nSonuç klasörü: {output_directory}")

    if not summary["all_passed"]:
        raise RuntimeError(
            "En az bir hareket doğrulaması başarısız."
        )


if __name__ == "__main__":
    main()