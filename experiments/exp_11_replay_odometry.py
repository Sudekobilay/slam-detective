"""Kaydedilmiş ölçümlerden hareket tahmini ve ayrı değerlendirme."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from estimation.encoder_gyro import EncoderGyroOdometry
from simulator.encoder import EncoderMeasurement
from simulator.imu import GyroMeasurement
from simulator.robot import Pose


def replay_measurements(input_path):
    """Yalnız ölçüm dosyasını okur; gerçek rotaya erişmez."""
    events = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))

    if not events:
        raise ValueError("Ölçüm dosyası boş.")

    seen = set()
    previous_time = -1.0

    for event in events:
        if event["schema_version"] != 1:
            raise ValueError("Desteklenmeyen kayıt sürümü.")

        time_s = event["time_s"]

        if not math.isfinite(time_s) or time_s < previous_time:
            raise ValueError("Ölçüm zamanları sıralı olmalıdır.")

        key = (
            event["sensor_type"],
            event["sensor_id"],
            time_s,
        )
        if key in seen:
            raise ValueError("Yinelenen sensör-zaman kaydı.")

        seen.add(key)
        previous_time = time_s

    # Bilinen başlangıç öncülü; gerçek rota dosyasından okunmaz.
    initial_pose = Pose(1.0, 3.0, 0.0)

    weights = {
        "encoder": 0.0,
        "encoder_gyro": 1.0,
        "blend_50": 0.5,
    }
    estimators = {
        name: EncoderGyroOdometry(
            gyro_weight=weight,
            initial_pose=initial_pose,
        )
        for name, weight in weights.items()
    }

    paths = {
        name: [{
            "time_s": 0.0,
            "x_m": initial_pose.x,
            "y_m": initial_pose.y,
            "theta_rad": initial_pose.theta,
        }]
        for name in estimators
    }

    update_count = 0

    for time_s, group in itertools.groupby(
        events, key=lambda event: event["time_s"]
    ):
        group = list(group)

        encoders = [
            event for event in group
            if event["sensor_type"] == "encoder"
        ]
        gyros = [
            event for event in group
            if event["sensor_type"] == "gyro"
        ]

        # GPS/LiDAR-only zamanlarında hareket güncellemesi yok.
        if not encoders and not gyros:
            continue

        if len(encoders) != 1 or len(gyros) != 1:
            raise ValueError(
                "Bu sürüm her aralıkta bir encoder "
                "ve bir gyro paketi gerektirir."
            )

        encoder = EncoderMeasurement(
            time_s=time_s,
            **encoders[0]["data"],
        )
        gyro = GyroMeasurement(
            time_s=time_s,
            **gyros[0]["data"],
        )

        for name, estimator in estimators.items():
            pose = estimator.update(encoder, gyro)

            paths[name].append({
                "time_s": time_s,
                "x_m": pose.x,
                "y_m": pose.y,
                "theta_rad": pose.theta,
            })

        update_count += 1

    if update_count == 0:
        raise ValueError("İşlenebilir hareket ölçümü bulunamadı.")

    return paths, weights, update_count


def evaluate(paths, truth_path):
    """Tahmin bittikten sonra gerçek rota ile hata hesaplar."""
    with truth_path.open("r", encoding="utf-8") as file:
        truth = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]

    truth_by_time = {
        round(row["time_s"], 12): row
        for row in truth
    }

    results = {}

    for name, path in paths.items():
        rows = []

        for estimate in path:
            time_s = round(estimate["time_s"], 12)

            if time_s not in truth_by_time:
                raise ValueError("Gerçek rota zamanı eşleşmedi.")

            actual = truth_by_time[time_s]

            position_error = math.hypot(
                estimate["x_m"] - actual["x_m"],
                estimate["y_m"] - actual["y_m"],
            )
            difference = estimate["theta_rad"] - actual["theta_rad"]
            heading_error = math.atan2(
                math.sin(difference), math.cos(difference)
            )

            rows.append({
                **estimate,
                "position_error_m": position_error,
                "heading_error_rad": heading_error,
            })

        results[name] = rows

    return truth, results


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    args = parser.parse_args()

    # Önce tahmin. Bu fonksiyona truth yolu verilmez.
    paths, weights, update_count = replay_measurements(args.input)

    # Sonra çevrimdışı değerlendirme.
    truth, results = evaluate(paths, args.truth)

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_11_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    summary = {
        "experiment": "EXP-11: Encoder-gyro kayıt tekrarı",
        "run_id": run_id,
        "input_sha256": hashlib.sha256(
            args.input.read_bytes()
        ).hexdigest(),
        "truth_sha256": hashlib.sha256(
            args.truth.read_bytes()
        ).hexdigest(),
        "initial_pose_prior": {
            "x_m": 1.0, "y_m": 3.0, "theta_rad": 0.0,
        },
        "assumed_calibration": {
            "left_radius_m": 0.05,
            "right_radius_m": 0.05,
            "wheelbase_m": 0.30,
            "counts_per_revolution": 2048,
        },
        "update_count": update_count,
        "methods": {},
        "limitations": [
            "Başlangıç pozu ve nominal kalibrasyon biliniyor.",
            "Sabit ağırlıklı başlangıç yöntemi; EKF değil.",
            "GPS ve LiDAR tahminde kullanılmıyor.",
            "Eksik encoder/gyro aralığı desteklenmiyor.",
            "Bias düzeltmesi ve kovaryans hesabı yok.",
        ],
    }

    figure, axes = plt.subplots(
        1, 2, figsize=(13, 6), constrained_layout=True
    )

    for name, rows in results.items():
        write_csv(output / f"{name}_estimate.csv", paths[name])
        write_csv(output / f"{name}_evaluation.csv", rows)

        position_rmse = math.sqrt(
            sum(row["position_error_m"]**2 for row in rows)
            / len(rows)
        )
        heading_rmse = math.sqrt(
            sum(row["heading_error_rad"]**2 for row in rows)
            / len(rows)
        )

        summary["methods"][name] = {
            "gyro_weight": weights[name],
            "position_rmse_m": position_rmse,
            "heading_rmse_rad": heading_rmse,
            "sample_count": len(rows),
        }

        axes[0].plot(
            [row["x_m"] for row in rows],
            [row["y_m"] for row in rows],
            label=name,
        )
        axes[1].plot(
            [row["time_s"] for row in rows],
            [row["position_error_m"] for row in rows],
            label=name,
        )

        print(
            f"{name}: konum RMSE={position_rmse:.6f} m | "
            f"yön RMSE={heading_rmse:.6f} rad"
        )

    axes[0].plot(
        [row["x_m"] for row in truth],
        [row["y_m"] for row in truth],
        "k--", label="Gerçek rota",
    )
    axes[0].set(
        title="Aynı kayıttan rota tahminleri",
        xlabel="x (m)", ylabel="y (m)",
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(
        title="Konum hatası",
        xlabel="Zaman (s)", ylabel="Hata (m)",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.savefig(output / "replay_comparison.png", dpi=160)
    plt.close(figure)

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    print(f"İşlenen hareket aralığı: {update_count}")
    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()