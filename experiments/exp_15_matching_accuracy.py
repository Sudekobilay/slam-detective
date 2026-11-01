"""Kaydedilmiş ICP sonuçlarının gerçek göreli hareketle değerlendirilmesi."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.exp_14_lidar_matching import relative_sensor_transform
from simulator.mounting import SensorMount
from simulator.robot import Pose, wrap_angle


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def rms(values):
    return math.sqrt(
        sum(value * value for value in values) / len(values)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    args = parser.parse_args()

    measurement_path = args.recording / "measurements.jsonl"
    truth_path = args.recording / "ground_truth.csv"
    manifest_path = args.recording / "summary.json"

    source_paths = {
        "matching": args.matching,
        "measurements": measurement_path,
        "ground_truth": truth_path,
        "recording_manifest": manifest_path,
    }
    original_hashes = {
        name: file_hash(path)
        for name, path in source_paths.items()
    }

    with args.matching.open("r", encoding="utf-8") as file:
        matching = json.load(file)

    # Yanlış kaydın gerçek rotasıyla değerlendirmeyi önle.
    if (
        matching["input_sha256"]
        != original_hashes["measurements"]
    ):
        raise ValueError(
            "ICP sonuçları seçilen ölçüm kaydına ait değil."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    # Gerçek montajlar yalnız bu değerlendirme kodunda okunur.
    actual_mounts = {
        unit["sensor_id"]: SensorMount(**unit["actual_mount"])
        for unit in manifest["lidar_units"]
    }

    with truth_path.open("r", encoding="utf-8") as file:
        truth_rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]

    truth_by_time = {}

    for row in truth_rows:
        time_s = round(row["time_s"], 12)

        if time_s in truth_by_time:
            raise ValueError("Gerçek rotada yinelenen zaman.")

        truth_by_time[time_s] = Pose(
            row["x_m"],
            row["y_m"],
            row["theta_rad"],
        )

    rows = []

    for item in matching["results"]:
        sensor_id = item["sensor_id"]

        if sensor_id not in actual_mounts:
            raise ValueError("Gerçek sensör montajı bulunamadı.")

        previous_time = round(item["previous_time_s"], 12)
        current_time = round(item["time_s"], 12)

        if current_time <= previous_time:
            raise ValueError("Geçersiz tarama zaman aralığı.")

        if (
            previous_time not in truth_by_time
            or current_time not in truth_by_time
        ):
            raise ValueError("Tarama zamanları gerçek rotada yok.")

        mount = actual_mounts[sensor_id]

        previous_sensor_pose = mount.world_pose(
            truth_by_time[previous_time]
        )
        current_sensor_pose = mount.world_pose(
            truth_by_time[current_time]
        )

        true_translation, true_yaw = relative_sensor_transform(
            previous_sensor_pose,
            current_sensor_pose,
        )

        result = item["result"]

        icp_translation_error = math.dist(
            result["translation_m"],
            true_translation,
        )
        icp_yaw = math.atan2(
            result["rotation"][1][0],
            result["rotation"][0][0],
        )
        icp_yaw_error = wrap_angle(icp_yaw - true_yaw)

        encoder_translation_error = math.dist(
            item["initial_translation_m"],
            true_translation,
        )
        encoder_yaw_error = wrap_angle(
            item["predicted_yaw_rad"] - true_yaw
        )

        rows.append({
            "sensor_id": sensor_id,
            "previous_time_s": previous_time,
            "time_s": current_time,
            "converged": result["converged"],
            "reason": result["reason"],
            "fit_rmse_m": result["rmse_m"],
            "true_translation_x_m": true_translation[0],
            "true_translation_y_m": true_translation[1],
            "true_yaw_rad": true_yaw,
            "icp_translation_error_m": icp_translation_error,
            "encoder_translation_error_m": encoder_translation_error,
            "icp_yaw_error_rad": icp_yaw_error,
            "encoder_yaw_error_rad": encoder_yaw_error,
        })

    if not rows:
        raise ValueError("Değerlendirilecek tarama çifti yok.")

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_15_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    write_csv(output / "relative_motion_errors.csv", rows)

    sensor_ids = sorted({
        row["sensor_id"] for row in rows
    })

    figure, axes = plt.subplots(
        2,
        len(sensor_ids),
        figsize=(7 * len(sensor_ids), 9),
        constrained_layout=True,
        squeeze=False,
    )

    statistics = {}

    for column, sensor_id in enumerate(sensor_ids):
        sensor_rows = [
            row for row in rows
            if row["sensor_id"] == sensor_id
        ]
        usable = [
            row for row in sensor_rows
            if row["converged"]
        ]

        statistics[sensor_id] = {
            "total_pairs": len(sensor_rows),
            "converged_pairs": len(usable),
            "failed_pairs": len(sensor_rows) - len(usable),
        }

        if not usable:
            print(
                f"{sensor_id}: yakınsayan çift yok; "
                "başarı metrikleri hesaplanmadı."
            )
            continue

        # İki yöntem aynı yakınsayan aralıklar üzerinde kıyaslanır.
        for method in ("encoder", "icp"):
            translation_rmse = rms([
                row[f"{method}_translation_error_m"]
                for row in usable
            ])
            yaw_rmse = rms([
                row[f"{method}_yaw_error_rad"]
                for row in usable
            ])

            statistics[sensor_id][method] = {
                "relative_translation_rmse_m": translation_rmse,
                "relative_yaw_rmse_rad": yaw_rmse,
            }

            axes[0, column].plot(
                [row["time_s"] for row in usable],
                [
                    row[f"{method}_translation_error_m"]
                    for row in usable
                ],
                marker=".",
                label=method,
            )
            axes[1, column].plot(
                [row["time_s"] for row in usable],
                [
                    math.degrees(row[f"{method}_yaw_error_rad"])
                    for row in usable
                ],
                marker=".",
                label=method,
            )

            print(
                f"{sensor_id}/{method}: "
                f"göreli öteleme RMSE={translation_rmse:.6f} m | "
                f"göreli yön RMSE={yaw_rmse:.6f} rad"
            )

        axes[0, column].set(
            title=f"{sensor_id} — Göreli öteleme hatası",
            xlabel="Aralık bitiş zamanı (s)",
            ylabel="Hata (m)",
        )
        axes[1, column].set(
            title=f"{sensor_id} — İşaretli göreli yön hatası",
            xlabel="Aralık bitiş zamanı (s)",
            ylabel="Yön hatası (derece)",
        )

        for row_index in (0, 1):
            axes[row_index, column].grid(alpha=0.25)
            axes[row_index, column].legend()

    figure.suptitle(
        "SLAM-Detective — Tarama Uyumu ve Gerçek Hareket",
        fontsize=15,
    )
    figure.savefig(
        output / "matching_accuracy.png",
        dpi=160,
    )
    plt.close(figure)

    source_files_unchanged = all(
        file_hash(path) == original_hashes[name]
        for name, path in source_paths.items()
    )

    summary = {
        "experiment": "EXP-15: ICP göreli hareket doğruluğu",
        "run_id": run_id,
        "source_hashes": original_hashes,
        "source_files_unchanged": source_files_unchanged,
        "pair_count": len(rows),
        "statistics": statistics,
        "limitations": [
            "Gerçek poz ve montaj yalnız çevrimdışı değerlendirmede.",
            "ICP yeniden çalıştırılmadı veya ayarlanmadı.",
            "Hatalar önceki sensör koordinatındaki göreli "
            "dönüşümler için hesaplanır; global rota ATE'si değildir.",
            "Encoder ve ICP aynı yakınsayan aralıklarda karşılaştırılır.",
            "Başarısız sonuçlar CSV'de ve sayımlarda korunur.",
            "ICP encoder başlangıç tahmininden yararlanır; "
            "tamamen bağımsız bir kestirim değildir.",
            "Tek kısa rota; genel yöntem sıralaması değildir.",
        ],
    }

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"Değerlendirilen çift: {len(rows)}")
    print(
        "Kaynak dosyaları korunmuş: "
        + ("PASS" if source_files_unchanged else "FAIL")
    )
    print(f"Sonuç klasörü: {output}")

    if not source_files_unchanged:
        raise RuntimeError("Kaynak dosyalardan biri değişmiş.")


if __name__ == "__main__":
    main()