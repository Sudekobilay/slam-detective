"""EXP-33 kaydından pasif/adaptif hareket GIF'i üretir.

Gerçek poz ve ortam yalnız çizimde kullanılır.
Denetleyiciye sadece ham ölçümler verilir.
"""

import argparse
import csv
from datetime import datetime, timezone
from itertools import groupby
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import (
    FuncAnimation,
    PillowWriter,
)

from estimation.hypothesis_motion import (
    HypothesisMotionAgent,
)
from estimation.scan_alignment_guard import (
    ScanGuardConfig,
)
from experiments.exp_24_motion_comparison import (
    write_json,
)
from simulator.environment import (
    create_demo_environment,
)


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return [
            {
                key: float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def collect_frames(
    folder,
    config,
    normal,
    policy,
):
    truth = {
        round(row["time_s"], 9): row
        for row in read_csv(
            folder / "ground_truth.csv"
        )
    }

    saved = {
        round(row["time_s"], 9): row
        for row in read_csv(
            folder / "line_slam.csv"
        )
    }

    events = [
        json.loads(line)
        for line in (
            folder / "measurements.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    agent = HypothesisMotionAgent(
        policy,
        normal,
        config,
    )

    frames = []
    points = {}
    origins = {}

    for time, batch in groupby(
        events,
        key=lambda event: event["time_s"],
    ):
        batch = list(batch)
        agent.process(batch)

        pose = agent.estimator.pose

        for event in batch:
            if event["sensor_type"] == "lidar":
                sensor = event["sensor_id"]
                mount = agent.mounts[sensor]
                sensor_pose = mount.world_pose(
                    pose
                )

                origins[sensor] = (
                    sensor_pose.x,
                    sensor_pose.y,
                )

                points[sensor] = [
                    mount.point_to_world(
                        point,
                        pose,
                    )
                    for point in event[
                        "data"
                    ]["scan"]["points"]
                    if point is not None
                ]

        expected = saved[round(time, 9)]

        if max(
            abs(pose.x - expected["x_m"]),
            abs(pose.y - expected["y_m"]),
        ) > 1e-7:
            raise RuntimeError(
                "Yeniden oynatma, kayıtlı "
                "kestirimle uyuşmuyor."
            )

        frames.append({
            "time_s": time,
            "truth": truth[round(time, 9)],
            "estimate": (
                pose.x,
                pose.y,
            ),
            "mode": agent.mode,
            "alarm": agent.hold.active,
            "motion": agent.selected_motion,
            "points": dict(points),
            "origins": dict(origins),
            "map": [
                (
                    feature.start,
                    feature.end,
                )
                for feature
                in agent.estimator.map_features()
            ],
        })

    return frames


def choose_case(
    run,
    normal,
    yaw,
    seed,
):
    candidates = sorted(
        (
            run / f"normal_{normal}"
        ).glob(
            f"seed_*/yaw_{yaw:g}/"
            "adaptive/summary.json"
        )
    )

    if seed is not None:
        candidates = [
            path
            for path in candidates
            if (
                path.parent.parent.parent.name
                == f"seed_{seed}"
            )
        ]

    if not candidates:
        raise ValueError(
            "İstenen EXP-33 koşulu bulunamadı."
        )

    # Yalnız gösterim için, komut değiştiren
    # ilk örnek seçilir. Nicel sonuçlar değişmez.
    for path in candidates:
        row = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            row.get(
                "changed_command_duration_s"
            ) or 0
        ) > 1e-9:
            return path.parent.parent

    return candidates[0].parent.parent


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--normal",
        choices=["straight", "arc"],
        default="straight",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
    )

    args = parser.parse_args()

    summary = json.loads(
        (
            args.run / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        summary.get("experiment_id")
        != "exp_33"
    ):
        parser.error(
            "--run bir EXP-33 sonuç "
            "klasörü olmalı."
        )

    calibration = json.loads(
        (
            args.run / "calibration.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    config = ScanGuardConfig(
        **calibration["config"]
    )

    case = choose_case(
        args.run,
        args.normal,
        args.yaw,
        args.seed,
    )

    print(
        f"Gösterilen kayıt: {case}",
        flush=True,
    )

    streams = {
        policy: collect_frames(
            case / policy,
            config,
            args.normal,
            policy,
        )
        for policy in (
            "passive",
            "adaptive",
        )
    }

    if (
        len(streams["passive"])
        != len(streams["adaptive"])
        or any(
            abs(
                first["time_s"]
                - second["time_s"]
            ) > 1e-9
            for first, second in zip(
                streams["passive"],
                streams["adaptive"],
            )
        )
    ):
        raise RuntimeError(
            "Gösterim zamanları eşleşmiyor."
        )

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = (
        root / "results" / f"demo_{stamp}"
    )
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
        constrained_layout=True,
    )

    environment = create_demo_environment()

    all_truth = [
        row["truth"]
        for stream in streams.values()
        for row in stream
    ]

    xmin = min(
        row["x_m"] for row in all_truth
    ) - 0.55
    xmax = max(
        row["x_m"] for row in all_truth
    ) + 0.55
    ymin = min(
        row["y_m"] for row in all_truth
    ) - 0.55
    ymax = max(
        row["y_m"] for row in all_truth
    ) + 0.55

    labels = {
        "straight": "Düz",
        "arc": "Yay",
        None: "Normal rota",
    }

    modes = {
        "NORMAL": "Normal sürüş",
        "WATCH": "Kontrol ediliyor",
        "PASSIVE_HOLD": "Normal rotada gözlem",
        "DIAGNOSING": "Teşhis hareketi",
        "DIAGNOSIS_WINDOW_ENDED": (
            "Gözlem tamamlandı"
        ),
        "NO_SUPPORTED_MANEUVER": (
            "Uygun hareket yok"
        ),
    }

    colors = {
        "passive": "darkorange",
        "adaptive": "royalblue",
    }

    def draw(index):
        for axis in axes:
            axis.clear()
            axis.set_aspect("equal")
            axis.grid(alpha=0.2)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")

            for wall in environment.walls:
                axis.plot(
                    [
                        wall.start[0],
                        wall.end[0],
                    ],
                    [
                        wall.start[1],
                        wall.end[1],
                    ],
                    color="black",
                    lw=1.0,
                )

        axes[0].set(
            xlim=(-0.3, 10.3),
            ylim=(-0.3, 8.3),
            title="Ortam ve gerçek rotalar",
        )

        for line_index, (start, end) in enumerate(
            streams["adaptive"][index]["map"]
        ):
            axes[0].plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="mediumpurple",
                alpha=0.65,
                lw=2,
                label=(
                    "Öğrenilen harita"
                    if line_index == 0
                    else None
                ),
            )

        for panel, policy in enumerate(
            ("passive", "adaptive"),
            start=1,
        ):
            stream = streams[policy]
            frame = stream[index]
            prefix = stream[:index + 1]

            x = [
                row["truth"]["x_m"]
                for row in prefix
            ]
            y = [
                row["truth"]["y_m"]
                for row in prefix
            ]

            axes[0].plot(
                x,
                y,
                color=colors[policy],
                label=(
                    "Pasif"
                    if policy == "passive"
                    else "Adaptif"
                ),
            )
            axes[0].scatter(
                x[-1],
                y[-1],
                color=colors[policy],
                s=30,
            )

            axis = axes[panel]
            axis.set(
                xlim=(xmin, xmax),
                ylim=(ymin, ymax),
            )

            for start, end in frame["map"]:
                axis.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color="mediumpurple",
                    alpha=0.55,
                    lw=1.0,
                )

            for sensor, cloud in (
                frame["points"].items()
            ):
                if cloud:
                    origin = frame[
                        "origins"
                    ][sensor]

                    color = (
                        "teal"
                        if sensor == "rear"
                        else "crimson"
                    )

                    for point in cloud[::12]:
                        axis.plot(
                            [origin[0], point[0]],
                            [origin[1], point[1]],
                            color=color,
                            alpha=0.12,
                            lw=0.7,
                        )

                    axis.scatter(
                        [
                            point[0]
                            for point in cloud
                        ],
                        [
                            point[1]
                            for point in cloud
                        ],
                        s=3,
                        alpha=0.3,
                        color=color,
                    )

            axis.plot(
                x,
                y,
                color="darkorange",
                lw=2,
                label="Gerçek rota",
            )

            axis.plot(
                [
                    row["estimate"][0]
                    for row in prefix
                ],
                [
                    row["estimate"][1]
                    for row in prefix
                ],
                color="royalblue",
                ls="--",
                label="SLAM tahmini",
            )

            truth = frame["truth"]

            axis.add_patch(
                plt.Circle(
                    (
                        truth["x_m"],
                        truth["y_m"],
                    ),
                    0.035,
                    color=colors[policy],
                )
            )

            axis.arrow(
                truth["x_m"],
                truth["y_m"],
                0.12 * math.cos(
                    truth["theta_rad"]
                ),
                0.12 * math.sin(
                    truth["theta_rad"]
                ),
                head_width=0.035,
                color=colors[policy],
                length_includes_head=True,
            )

            title = (
                "Pasif"
                if policy == "passive"
                else "Adaptif"
            )

            alarm = (
                "Var"
                if frame["alarm"]
                else "Yok"
            )

            mode = modes.get(
                frame["mode"],
                frame["mode"],
            )

            axis.set_title(
                f"{title} | Alarm: {alarm}\n"
                f"Hareket: {labels[frame['motion']]} | "
                f"{mode}",
                fontsize=10,
            )

            axis.legend(
                loc="lower right",
                fontsize=8,
            )

        axes[0].legend(
            loc="upper right",
            fontsize=8,
        )

        time = streams[
            "adaptive"
        ][index]["time_s"]

        figure.suptitle(
            f"SLAM-Detective | t={time:.1f} s | "
            f"Ön LiDAR ofseti: {args.yaw:g}°\n"
            "Kayıtlı çevrim içi deneyin yeniden "
            "gösterimi; gerçek bilgiler yalnız çizimdedir.",
            fontsize=11,
        )

    count = len(streams["adaptive"])

    animation = FuncAnimation(
        figure,
        draw,
        frames=count,
        interval=100,
        blit=False,
    )

    animation.save(
        output / "motion_demo.gif",
        writer=PillowWriter(fps=10),
        dpi=100,
    )

    draw(count - 1)

    figure.savefig(
        output / "final_frame.png",
        dpi=140,
    )
    plt.close(figure)

    write_json(
        output / "summary.json",
        {
            "source_run": str(args.run),
            "source_case": str(case),
            "frame_count": count,
            "replayed_estimates_match_saved": True,
            "selection_note": (
                "Komut değiştiren ilk örnek yalnız "
                "gösterim için seçilir; böyle örnek "
                "yoksa ilk kayıt gösterilir."
            ),
            "limitations": [
                "Bu animasyon EXP-33 EKF-SLAM "
                "deneyini gösterir; FastSLAM veya "
                "GPS alarmıyla hareket seçimi "
                "gösterimi değildir.",
                "LiDAR noktaları son taramadan "
                "bir sonraki taramaya kadar tutulur; "
                "her animasyon karesinde yeni "
                "tarama varmış gibi yorumlanmaz.",
            ],
        },
    )

    print(
        "Yeniden oynatma doğrulaması: PASS"
    )
    print(
        f"Hareketli gösterim: "
        f"{output / 'motion_demo.gif'}"
    )
    print(
        f"Son kare: "
        f"{output / 'final_frame.png'}"
    )


if __name__ == "__main__":
    main()