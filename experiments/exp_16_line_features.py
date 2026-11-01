"""Kaydedilmiş gerçek LiDAR ölçüm hattından doğru özellikleri çıkarır."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from perception.line_features import LineConfig, extract_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    events = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    config = LineConfig()
    feature_records = []
    last_scans = {}
    statistics = defaultdict(
        lambda: {"scans": 0, "features": 0}
    )

    for event in events:
        if event["sensor_type"] != "lidar":
            continue

        sensor_id = event["sensor_id"]
        points = event["data"]["scan"]["points"]
        lines = extract_lines(points, config)

        record = {
            "sensor_id": sensor_id,
            "time_s": event["time_s"],
            "features": [asdict(line) for line in lines],
        }
        feature_records.append(record)

        statistics[sensor_id]["scans"] += 1
        statistics[sensor_id]["features"] += len(lines)
        last_scans[sensor_id] = {
            "time_s": event["time_s"],
            "points": points,
            "lines": lines,
        }

    if not feature_records:
        raise ValueError("LiDAR kaydı bulunamadı.")

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_16_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    with (output / "line_features.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for record in feature_records:
            file.write(json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n")

    sensor_ids = sorted(last_scans)

    figure, axes = plt.subplots(
        1,
        len(sensor_ids),
        figsize=(7 * len(sensor_ids), 6),
        constrained_layout=True,
        squeeze=False,
    )

    for column, sensor_id in enumerate(sensor_ids):
        axis = axes[0, column]
        scan = last_scans[sensor_id]
        points = [
            point for point in scan["points"]
            if point is not None
        ]

        if points:
            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                color="gray",
                s=15,
                label="Ham tarama noktaları",
            )

        for index, line in enumerate(scan["lines"]):
            axis.plot(
                [line.start[0], line.end[0]],
                [line.start[1], line.end[1]],
                linewidth=2.5,
                label="Çıkarılan parçalar" if index == 0 else None,
            )

        axis.scatter(
            0.0, 0.0,
            color="black",
            s=45,
            label="Sensör merkezi",
        )

        axis.set(
            title=(
                f"{sensor_id} — t={scan['time_s']:.1f} s\n"
                f"{len(scan['lines'])} doğru parçası"
            ),
            xlabel="Sensör x (m)",
            ylabel="Sensör y (m)",
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

        stats = statistics[sensor_id]
        print(
            f"{sensor_id}: tarama={stats['scans']} | "
            f"toplam özellik={stats['features']}"
        )

    all_features = [
        feature
        for record in feature_records
        for feature in record["features"]
    ]

    checks = {
        "features_produced": len(all_features) > 0,
        "support_and_length": all(
            feature["support_count"] >= config.min_points
            and feature["length_m"] >= config.min_length_m
            for feature in all_features
        ),
        "fitting_tolerance": all(
            feature["max_point_error_m"]
            <= config.max_point_error_m + 1e-12
            for feature in all_features
        ),
        "unit_normals": all(
            math.isclose(
                math.hypot(*feature["normal"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for feature in all_features
        ),
        "source_unchanged": args.input.read_bytes() == source_bytes,
    }

    summary = {
        "experiment": "EXP-16: Ham taramadan doğru özellikleri",
        "run_id": run_id,
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "config": asdict(config),
        "statistics": dict(statistics),
        "checks": checks,
        "all_passed": all(checks.values()),
        "limitations": [
            "Yalnız sensör koordinatında özellik çıkarımı.",
            "Gerçek poz veya kusursuz ortam haritası okunmuyor.",
            "Özelliklere gerçek duvar kimliği verilmiyor.",
            "Parça birleştirme ve taramalar arası ilişkilendirme yok.",
            "İlk ve son ışınlar arasında çevrimsel birleştirme yok.",
            "Eşik altı artık, doğru fiziksel duvar bulunduğunun "
            "tek başına kanıtı değildir.",
            "Haritalama ve SLAM henüz yapılmıyor.",
        ],
    }

    figure.suptitle(
        "SLAM-Detective — Ham LiDAR'dan Duvar Özellikleri",
        fontsize=14,
    )
    figure.savefig(output / "line_features.png", dpi=160)
    plt.close(figure)

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"Sonuç klasörü: {output}")

    if not summary["all_passed"]:
        raise RuntimeError("Özellik çıkarımı kontrolleri başarısız.")


if __name__ == "__main__":
    main()