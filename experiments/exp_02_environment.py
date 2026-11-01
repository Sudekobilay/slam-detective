"""İlk ortamın geometrisini kontrol eder ve görselleştirir."""

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
from simulator.robot import Pose


def main():
    environment = create_demo_environment()
    robot_pose = Pose(1.0, 1.0, 0.0)

    checks = {
        "wall_count": len(environment.walls) == 12,
        "outer_perimeter": math.isclose(
            sum(
                wall.length
                for wall in environment.boundary.walls()
            ),
            36.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "total_wall_length": math.isclose(
            sum(wall.length for wall in environment.walls),
            49.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError(
            "Ortamın geometrik kontrolleri başarısız."
        )

    project_root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_directory = (
        project_root / "results" / f"exp_02_{run_id}"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    figure, axis = plt.subplots(
        figsize=(9, 7),
        constrained_layout=True,
    )

    # Engel içlerini dolu bölge olarak göster.
    for index, obstacle in enumerate(
        environment.obstacles, start=1
    ):
        axis.add_patch(
            PlotRectangle(
                (obstacle.x_min, obstacle.y_min),
                obstacle.x_max - obstacle.x_min,
                obstacle.y_max - obstacle.y_min,
                facecolor="lightgray",
                edgecolor="none",
            )
        )

        axis.text(
            (obstacle.x_min + obstacle.x_max) / 2,
            (obstacle.y_min + obstacle.y_max) / 2,
            f"Engel {index}",
            ha="center",
            va="center",
        )

    for wall in environment.walls:
        axis.plot(
            [wall.start[0], wall.end[0]],
            [wall.start[1], wall.end[1]],
            color="black",
            linewidth=2,
        )

    axis.scatter(
        robot_pose.x,
        robot_pose.y,
        color="royalblue",
        s=90,
        label="Robot merkezi",
        zorder=4,
    )

    axis.arrow(
        robot_pose.x,
        robot_pose.y,
        0.6 * math.cos(robot_pose.theta),
        0.6 * math.sin(robot_pose.theta),
        head_width=0.15,
        length_includes_head=True,
        color="royalblue",
        zorder=4,
    )

    axis.set(
        title="SLAM-Detective — İlk 2B Ortam",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(-0.5, 10.5),
        ylim=(-0.5, 8.5),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.savefig(
        output_directory / "environment.png",
        dpi=160,
    )
    plt.close(figure)

    record = {
        "experiment": "EXP-02: Ortam geometrisi",
        "run_id": run_id,
        "length_unit": "m",
        "boundary": asdict(environment.boundary),
        "obstacles": [
            asdict(obstacle)
            for obstacle in environment.obstacles
        ],
        "walls": [
            asdict(wall)
            for wall in environment.walls
        ],
        "robot_pose": asdict(robot_pose),
        "wall_count": len(environment.walls),
        "total_wall_length_m": sum(
            wall.length for wall in environment.walls
        ),
        "checks": checks,
        "all_passed": all(checks.values()),
        "limitations": [
            "Robot bu deneyde hareket etmiyor.",
            "LiDAR henüz uygulanmadı.",
            "Robot gövdesi ve çarpışma kontrolü yok.",
            "Engellerin birbirleriyle örtüşmesi "
            "genel olarak kontrol edilmiyor.",
        ],
    }

    with (output_directory / "environment.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"Duvar sayısı: {len(environment.walls)}")
    print(f"Sonuç klasörü: {output_directory}")


if __name__ == "__main__":
    main()