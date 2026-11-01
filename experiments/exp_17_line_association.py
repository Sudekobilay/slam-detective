"""Ham özellikleri encoder pozuyla sabit ilk-tarama referansına bağlar."""

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.exp_11_replay_odometry import replay_measurements
from perception.line_association import (
    AssociationConfig,
    associate_lines,
    transform_line,
)
from perception.line_features import LineFeature
from simulator.mounting import SensorMount
from simulator.robot import Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    feature_bytes = args.features.read_bytes()

    with (args.features.parent / "summary.json").open(
        "r", encoding="utf-8"
    ) as file:
        feature_summary = json.load(file)

    input_hash = hashlib.sha256(input_bytes).hexdigest()

    if feature_summary["input_sha256"] != input_hash:
        raise ValueError("Özellikler seçilen ölçüm kaydına ait değil.")

    paths, _, _ = replay_measurements(args.input)
    poses = {
        round(row["time_s"], 12): Pose(
            row["x_m"], row["y_m"], row["theta_rad"]
        )
        for row in paths["encoder"]
    }

    mounts = {
        "front": SensorMount(x=0.25),
        "rear": SensorMount(x=-0.25, yaw=math.pi),
    }
    config = AssociationConfig()

    records = [
        json.loads(line)
        for line in feature_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    references = {}
    last_frames = {}
    output_records = []
    counts = {sid: Counter() for sid in mounts}
    seen = set()
    previous_time = -1.0

    for record in records:
        sensor_id = record["sensor_id"]
        time_s = round(record["time_s"], 12)

        if sensor_id not in mounts or time_s not in poses:
            raise ValueError("Sensör veya zaman eşleşmedi.")

        if time_s < previous_time:
            raise ValueError("Özellik kayıtları sıralı olmalıdır.")

        if (sensor_id, time_s) in seen:
            raise ValueError("Yinelenen sensör-zaman kaydı.")

        seen.add((sensor_id, time_s))
        previous_time = time_s

        sensor_pose = mounts[sensor_id].world_pose(poses[time_s])

        world_lines = [
            transform_line(
                LineFeature(**feature),
                sensor_pose,
            )
            for feature in record["features"]
        ]

        if sensor_id not in references:
            if time_s != 0.0:
                raise ValueError("İlk referans t=0 taraması olmalıdır.")

            references[sensor_id] = world_lines
            matches = [{
                "observation_index": index,
                "reference_index": index,
                "status": "reference",
                "score": None,
            } for index in range(len(world_lines))]
        else:
            matches = associate_lines(
                world_lines,
                references[sensor_id],
                config,
            )
            counts[sensor_id].update(
                match["status"] for match in matches
            )

        frame = {
            "sensor_id": sensor_id,
            "time_s": time_s,
            "world_lines": [
                asdict(line) for line in world_lines
            ],
            "associations": matches,
        }

        output_records.append(frame)
        last_frames[sensor_id] = frame

    if not output_records:
        raise ValueError("Özellik kaydı boş.")

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_17_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    with (output / "associations.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for record in output_records:
            file.write(json.dumps(
                record, ensure_ascii=False, allow_nan=False
            ) + "\n")

    sensor_ids = sorted(last_frames)
    figure, axes = plt.subplots(
        1, len(sensor_ids),
        figsize=(7 * len(sensor_ids), 6),
        constrained_layout=True,
        squeeze=False,
    )

    colors = {
        "matched": "green",
        "ambiguous": "darkorange",
        "conflict": "crimson",
        "no_candidate": "gray",
        "reference": "royalblue",
    }

    for column, sensor_id in enumerate(sensor_ids):
        axis = axes[0, column]

        for index, line in enumerate(references[sensor_id]):
            axis.plot(
                [line.start[0], line.end[0]],
                [line.start[1], line.end[1]],
                color="black",
                linewidth=3,
                alpha=0.4,
                label="İlk tarama referansı" if index == 0 else None,
            )

        frame = last_frames[sensor_id]
        labels_used = set()

        for line, match in zip(
            frame["world_lines"],
            frame["associations"],
            strict=True,
        ):
            status = match["status"]

            axis.plot(
                [line["start"][0], line["end"][0]],
                [line["start"][1], line["end"][1]],
                color=colors[status],
                linewidth=1.8,
                label=status if status not in labels_used else None,
            )
            labels_used.add(status)

        axis.set(
            title=f"{sensor_id} — t={frame['time_s']:.1f} s",
            xlabel="Tahmini dünya x (m)",
            ylabel="Tahmini dünya y (m)",
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

        print(f"{sensor_id}: {dict(counts[sensor_id])}")

    figure.suptitle(
        "SLAM-Detective — Sabit Referansla Özellik Eşleştirme",
        fontsize=14,
    )
    figure.savefig(output / "line_association.png", dpi=160)
    plt.close(figure)

    unchanged = (
        args.input.read_bytes() == input_bytes
        and args.features.read_bytes() == feature_bytes
    )

    summary = {
        "experiment": "EXP-17: Doğru özelliklerini ilişkilendirme",
        "run_id": run_id,
        "input_sha256": input_hash,
        "features_sha256": hashlib.sha256(feature_bytes).hexdigest(),
        "association_config": asdict(config),
        "assumed_mounts": {
            sid: asdict(mount) for sid, mount in mounts.items()
        },
        "reference_lines": {
            sid: [asdict(line) for line in lines]
            for sid, lines in references.items()
        },
        "counts_excluding_reference": {
            sid: dict(counter) for sid, counter in counts.items()
        },
        "source_files_unchanged": unchanged,
        "limitations": [
            "Gerçek poz ve kusursuz ortam haritası okunmuyor.",
            "Koordinatlar encoder odometrisinden hesaplanıyor.",
            "Her sensör kendi ilk taramasıyla karşılaştırılıyor.",
            "Referans sabit; yeni yüzeyler haritaya eklenmiyor.",
            "Birleştirme, kovaryans ve ortak robot-harita "
            "güncellemesi yok; bu aşama SLAM değildir.",
            "Geometrik eşleşme fiziksel duvar kimliği kanıtı değildir.",
            "Eşleşmeme, sensör arızası anlamına gelmez.",
        ],
    }

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    print(
        "Kaynak dosyaları korunmuş: "
        + ("PASS" if unchanged else "FAIL")
    )
    print(f"Sonuç klasörü: {output}")

    if not unchanged:
        raise RuntimeError("Kaynak dosyalardan biri değişmiş.")


if __name__ == "__main__":
    main()