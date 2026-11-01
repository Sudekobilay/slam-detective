"""LiDAR montaj açısı hatasının nokta konumlarına etkisi."""

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
from simulator.mounting import SensorMount
from simulator.robot import Pose


def draw_environment(axis, environment):
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
            linewidth=1.5,
        )


def main():
    environment = create_demo_environment()
    robot_pose = Pose(1.0, 3.0, 0.0)

    config = LidarConfig(
        beam_count=180,
        max_range=15.0,
    )
    sensor = Lidar(config)

    # Yazılım her iki durumda da bu montajı kullanıyor.
    assumed_mount = SensorMount(
        x=0.25,
        y=0.0,
        yaw=0.0,
    )

    cases = (
        ("correct_mount", 0.0, "Doğru montaj bilgisi"),
        ("yaw_error_5deg", 5.0, "5° montaj açısı hatası"),
    )

    project_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_directory = (
        project_root / "results" / f"exp_04_{run_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    figure, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    summary = {
        "experiment": "EXP-04: Montaj açısı hatası",
        "run_id": run_id,
        "robot_pose": asdict(robot_pose),
        "lidar_config": asdict(config),
        "assumed_mount": asdict(assumed_mount),
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [
                asdict(obstacle)
                for obstacle in environment.obstacles
            ],
        },
        "cases": {},
        "scope": (
            "Sabit ve doğru bilinen robot pozu altında "
            "nokta geometrisi deneyi; SLAM veya teşhis değil."
        ),
    }

    all_checks = []

    for axis, (name, yaw_deg, title) in zip(axes, cases):
        actual_mount = SensorMount(
            x=0.25,
            y=0.0,
            yaw=math.radians(yaw_deg),
        )

        # Ölçümler gerçek fiziksel montajdan üretiliyor.
        scan = sensor.scan(
            actual_mount.world_pose(robot_pose),
            environment,
        )

        rows = []
        true_points = []
        reconstructed_points = []

        angle_error = (
            assumed_mount.yaw - actual_mount.yaw
        )

        for index, (distance, point) in enumerate(
            zip(scan.ranges, scan.points)
        ):
            if point is None:
                continue

            # Gerçek dönüşüm yalnız çevrimdışı değerlendirme.
            true_point = actual_mount.point_to_world(
                point, robot_pose
            )

            # Aynı ölçüm, yazılımın bildiği montajla yorumlanır.
            reconstructed = assumed_mount.point_to_world(
                point, robot_pose
            )

            error = math.dist(true_point, reconstructed)

            # Aynı sensör merkezinde yalnız açı farkı varsa:
            # hata = 2 × mesafe × |sin(açı_farkı / 2)|.
            expected_error = (
                2.0
                * distance
                * abs(math.sin(angle_error / 2.0))
            )

            true_points.append(true_point)
            reconstructed_points.append(reconstructed)

            rows.append({
                "beam_index": index,
                "range_m": distance,
                "true_world_x_m": true_point[0],
                "true_world_y_m": true_point[1],
                "reconstructed_world_x_m": reconstructed[0],
                "reconstructed_world_y_m": reconstructed[1],
                "point_error_m": error,
                "analytical_error_m": expected_error,
            })

        if not rows:
            raise RuntimeError("Geçerli LiDAR ölçümü yok.")

        errors = [row["point_error_m"] for row in rows]

        formula_matches = all(
            math.isclose(
                row["point_error_m"],
                row["analytical_error_m"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for row in rows
        )

        effect_matches = (
            max(errors) < 1e-9
            if yaw_deg == 0.0
            else max(errors) > 1e-3
        )

        passed = formula_matches and effect_matches
        all_checks.append(passed)

        metrics = {
            "valid_return_count": len(rows),
            "point_rmse_m": math.sqrt(
                sum(error**2 for error in errors)
                / len(errors)
            ),
            "mean_point_error_m": sum(errors) / len(errors),
            "maximum_point_error_m": max(errors),
            "analytical_formula_matches": formula_matches,
            "expected_effect_observed": effect_matches,
            "passed": passed,
        }

        summary["cases"][name] = {
            "actual_mount": asdict(actual_mount),
            "metrics": metrics,
        }

        with (output_directory / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        draw_environment(axis, environment)

        axis.scatter(
            [point[0] for point in true_points],
            [point[1] for point in true_points],
            s=18,
            facecolors="none",
            edgecolors="green",
            label="Gerçek yüzey noktaları",
            zorder=3,
        )

        axis.scatter(
            [point[0] for point in reconstructed_points],
            [point[1] for point in reconstructed_points],
            s=13,
            color="crimson",
            marker="x",
            label="Varsayılan montajla yerleştirilen",
            zorder=4,
        )

        axis.scatter(
            robot_pose.x,
            robot_pose.y,
            s=55,
            color="royalblue",
            label="Robot merkezi",
            zorder=5,
        )

        axis.set(
            title=(
                f"{title}\n"
                f"Nokta RMSE: {metrics['point_rmse_m']:.4f} m"
            ),
            xlabel="x (m)",
            ylabel="y (m)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="upper right")

        print(
            f"{name}: {'PASS' if passed else 'FAIL'} | "
            f"nokta RMSE={metrics['point_rmse_m']:.6f} m"
        )

    # İki paneli aynı ölçek ve sınırlarla karşılaştır.
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
        "SLAM-Detective — Gerçek ve Varsayılan Montaj",
        fontsize=15,
    )
    figure.savefig(
        output_directory / "mounting_error.png",
        dpi=160,
    )
    plt.close(figure)

    summary["all_passed"] = all(all_checks)

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
            "Montaj hatası deneyindeki kontroller başarısız."
        )


if __name__ == "__main__":
    main()