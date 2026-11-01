"""Kaydedilmiş LiDAR taramalarını sensör bazında eşleştirir."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from estimation.scan_matching import icp_2d
from experiments.exp_11_replay_odometry import (
    replay_measurements,
    write_csv,
)
from simulator.mounting import SensorMount
from simulator.robot import Pose, wrap_angle


def relative_sensor_transform(previous_pose, current_pose):
    """Şimdiki sensör koordinatından önceki sensöre dönüşüm."""
    dx = current_pose.x - previous_pose.x
    dy = current_pose.y - previous_pose.y

    cosine = math.cos(previous_pose.theta)
    sine = math.sin(previous_pose.theta)

    translation = (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
    )
    yaw = wrap_angle(
        current_pose.theta - previous_pose.theta
    )

    return translation, yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    # Gerçek rota dosyası bu deneye girdi değildir.
    paths, _, _ = replay_measurements(args.input)
    encoder_by_time = {
        round(row["time_s"], 12): Pose(
            row["x_m"], row["y_m"], row["theta_rad"]
        )
        for row in paths["encoder"]
    }

    assumed_mounts = {
        "front": SensorMount(x=0.25),
        "rear": SensorMount(x=-0.25, yaw=math.pi),
    }

    source_bytes = args.input.read_bytes()
    with args.input.open("r", encoding="utf-8") as file:
        events = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    previous = {}
    rows = []
    details = []

    for event in events:
        if event["sensor_type"] != "lidar":
            continue

        sensor_id = event["sensor_id"]

        if sensor_id not in assumed_mounts:
            raise ValueError("Sensörün nominal montajı tanımlı değil.")

        time_s = round(event["time_s"], 12)

        if time_s not in encoder_by_time:
            raise ValueError("LiDAR zamanı odometriyle eşleşmiyor.")

        points = [
            point
            for point in event["data"]["scan"]["points"]
            if point is not None
        ]
        sensor_pose = assumed_mounts[sensor_id].world_pose(
            encoder_by_time[time_s]
        )

        if sensor_id in previous:
            old = previous[sensor_id]

            predicted_translation, predicted_yaw = (
                relative_sensor_transform(
                    old["sensor_pose"], sensor_pose
                )
            )

            result = icp_2d(
                source=points,
                target=old["points"],
                initial_translation=predicted_translation,
                initial_yaw=predicted_yaw,
                max_distance=0.5,
                trim_fraction=0.85,
                min_pairs=6,
                max_iterations=60,
                tolerance=1e-7,
            )

            icp_yaw = math.atan2(
                result["rotation"][1][0],
                result["rotation"][0][0],
            )
            yaw_disagreement = wrap_angle(
                icp_yaw - predicted_yaw
            )

            row = {
                "sensor_id": sensor_id,
                "previous_time_s": old["time_s"],
                "time_s": time_s,
                "converged": result["converged"],
                "reason": result["reason"],
                "matched_points": result["matched_points"],
                "source_point_count": len(points),
                "iterations": result["iterations"],
                "fit_rmse_m": result["rmse_m"],
                "predicted_yaw_rad": predicted_yaw,
                "icp_yaw_rad": icp_yaw,
                "yaw_disagreement_rad": yaw_disagreement,
                "translation_disagreement_m": math.dist(
                    result["translation_m"],
                    predicted_translation,
                ),
            }

            rows.append(row)
            details.append({
                **row,
                "initial_translation_m": list(predicted_translation),
                "result": result,
            })

        previous[sensor_id] = {
            "time_s": time_s,
            "points": points,
            "sensor_pose": sensor_pose,
        }

    if not rows:
        raise ValueError("Eşleştirilebilir ardışık tarama bulunamadı.")

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_14_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    write_csv(output / "scan_matching.csv", rows)

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    for sensor_id in assumed_mounts:
        selected = [
            row for row in rows
            if row["sensor_id"] == sensor_id
        ]
        usable = [
            row for row in selected
            if row["converged"]
        ]

        axes[0].plot(
            [row["time_s"] for row in usable],
            [row["fit_rmse_m"] for row in usable],
            marker=".",
            label=sensor_id,
        )
        axes[1].plot(
            [row["time_s"] for row in usable],
            [
                math.degrees(row["yaw_disagreement_rad"])
                for row in usable
            ],
            marker=".",
            label=sensor_id,
        )

        print(
            f"{sensor_id}: "
            f"eşleşen={len(usable)}/{len(selected)}"
        )

    axes[0].set(
        title="Tutulan eşleşmelerin uyum hatası",
        xlabel="Zaman (s)", ylabel="ICP RMSE (m)",
    )
    axes[1].set(
        title="ICP ve encoder dönüşü arasındaki fark",
        xlabel="Zaman (s)", ylabel="Açı farkı (derece)",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()

    figure.savefig(
        output / "lidar_matching.png", dpi=160
    )
    plt.close(figure)

    summary = {
        "experiment": "EXP-14: LiDAR tarama eşleştirme",
        "run_id": run_id,
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "assumed_mounts": {
            key: {
                "x": mount.x,
                "y": mount.y,
                "yaw": mount.yaw,
            }
            for key, mount in assumed_mounts.items()
        },
        "icp_parameters": {
            "max_distance_m": 0.5,
            "trim_fraction": 0.85,
            "min_pairs": 6,
            "max_iterations": 60,
            "tolerance": 1e-7,
        },
        "pair_count": len(rows),
        "converged_count": sum(
            row["converged"] for row in rows
        ),
        "results": details,
        "source_file_unchanged": (
            args.input.read_bytes() == source_bytes
        ),
        "limitations": [
            "Gerçek rota ve kusursuz ortam haritası okunmuyor.",
            "Başlangıç dönüşümü encoder odometrisinden geliyor.",
            "Yalnız ardışık aynı-sensör taramaları eşleştiriliyor.",
            "Yakınsama, doğru hareket bulunduğunu garanti etmez.",
            "RMSE yalnız kapıdan ve kırpmadan geçen eşleşmelerindir.",
            "Başarısız eşleşmeler sonuç dosyasında korunur.",
            "Grafikler yalnız yakınsayan eşleşmeleri gösterir.",
            "Tam SLAM veya otomatik hata teşhisi değildir.",
        ],
    }

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    print(f"Toplam tarama çifti: {len(rows)}")
    print(
        "Yakınsayan çift: "
        f"{summary['converged_count']}/{len(rows)}"
    )
    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()