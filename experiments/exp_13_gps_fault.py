"""Aynı hareket kaydında nominal ve sapmalı GPS karşılaştırması."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.exp_11_replay_odometry import (
    evaluate,
    replay_measurements,
    write_csv,
)
from experiments.exp_12_pose_ekf import replay_ekf
from simulator.fault_injection import inject_gps_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    args = parser.parse_args()

    original_bytes = args.input.read_bytes()
    source_hash = hashlib.sha256(original_bytes).hexdigest()

    events = [
        json.loads(line)
        for line in original_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    start_s, end_s = 3.0, 4.1
    bias_x_m, bias_y_m = 0.5, -0.3

    faulty_events, affected_count = inject_gps_bias(
        events,
        start_s,
        end_s,
        bias_x_m,
        bias_y_m,
    )

    if affected_count == 0:
        raise ValueError(
            "Seçilen aralıkta geçerli GPS ölçümü bulunamadı."
        )

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_13_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    faulty_path = output / "faulty_measurements.jsonl"

    with faulty_path.open("w", encoding="utf-8") as file:
        for event in faulty_events:
            file.write(json.dumps(
                event,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n")

    # İki dosyanın da mevcut kayıt kontrollerinden geçmesi gerekir.
    replay_measurements(args.input)
    replay_measurements(faulty_path)

    nominal_paths, nominal_estimators = replay_ekf(args.input)
    faulty_paths, faulty_estimators = replay_ekf(faulty_path)

    paths = {
        "ekf_no_gps": nominal_paths["ekf_no_gps"],
        "ekf_nominal_gps": nominal_paths["ekf_gps"],
        "ekf_faulty_gps": faulty_paths["ekf_gps"],
    }

    # Gerçek rota ancak tahminlerden sonra değerlendirmeye girer.
    truth, evaluated = evaluate(paths, args.truth)

    checks = {
        "non_gps_measurements_unchanged": [
            event for event in events
            if event["sensor_type"] != "gps"
        ] == [
            event for event in faulty_events
            if event["sensor_type"] != "gps"
        ],
        "gps_availability_unchanged": [
            (event["time_s"], event["data"]["available"])
            for event in events
            if event["sensor_type"] == "gps"
        ] == [
            (event["time_s"], event["data"]["available"])
            for event in faulty_events
            if event["sensor_type"] == "gps"
        ],
        "no_gps_estimate_unchanged": (
            nominal_paths["ekf_no_gps"]
            == faulty_paths["ekf_no_gps"]
        ),
        "source_file_unchanged": (
            hashlib.sha256(args.input.read_bytes()).hexdigest()
            == source_hash
        ),
    }

    summary = {
        "experiment": "EXP-13: Bilinmeyen GPS konum sapması",
        "run_id": run_id,
        "source_sha256": source_hash,
        "faulty_stream_sha256": hashlib.sha256(
            faulty_path.read_bytes()
        ).hexdigest(),
        "truth_sha256": hashlib.sha256(
            args.truth.read_bytes()
        ).hexdigest(),
        "offline_injection_parameters": {
            "start_s": start_s,
            "end_s_exclusive": end_s,
            "bias_x_m": bias_x_m,
            "bias_y_m": bias_y_m,
            "affected_measurements": affected_count,
        },
        "filter_parameters": {
            "initial_pose": [1.0, 3.0, 0.0],
            "geometry": [0.05, 0.05, 0.30],
            "counts_per_revolution": 2048,
            "initial_covariance_diagonal": [
                1e-6, 1e-6, 0.005**2, 0.02**2
            ],
            "distance_noise_density": 0.005,
            "bias_random_walk": 0.001,
            "gps_variance_floor_m2": 1e-8,
        },
        "methods": {},
        "checks": checks,
        "limitations": [
            "Filtre parametreleri önceki deneyle aynı.",
            "Hata bilgisi yalnız enjeksiyon ve değerlendirmede.",
            "NIS kaydı var; eşik veya otomatik ret yok.",
            "LiDAR kestirime bağlı değil.",
            "Kayıt hata sürerken bitiyor; toparlanma ölçülmüyor.",
            "Tek kayıt; genel istatistiksel sonuç değildir.",
        ],
    }

    figure, axes = plt.subplots(
        1, 3, figsize=(17, 5), constrained_layout=True
    )

    for name, rows in evaluated.items():
        write_csv(output / f"{name}_estimate.csv", paths[name])
        write_csv(output / f"{name}_evaluation.csv", rows)

        rmse = math.sqrt(
            sum(row["position_error_m"]**2 for row in rows)
            / len(rows)
        )
        final_error = rows[-1]["position_error_m"]

        summary["methods"][name] = {
            "position_rmse_m": rmse,
            "final_position_error_m": final_error,
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
            f"{name}: RMSE={rmse:.6f} m | "
            f"son konum hatası={final_error:.6f} m"
        )

    for name, estimator in (
        ("nominal_gps", nominal_estimators["ekf_gps"]),
        ("faulty_gps", faulty_estimators["ekf_gps"]),
    ):
        innovations = estimator.innovations
        write_csv(output / f"{name}_innovations.csv", innovations)

        axes[2].plot(
            [row["time_s"] for row in innovations],
            [row["nis"] for row in innovations],
            marker=".",
            label=name,
        )

    axes[0].plot(
        [row["x_m"] for row in truth],
        [row["y_m"] for row in truth],
        "k--", label="Gerçek rota",
    )
    axes[0].set(
        title="Rota tahminleri",
        xlabel="x (m)", ylabel="y (m)",
    )
    axes[0].set_aspect("equal", adjustable="datalim")

    axes[1].set(
        title="Konum hatası",
        xlabel="Zaman (s)", ylabel="Hata (m)",
    )
    axes[2].set(
        title="GPS inovasyon tutarsızlığı",
        xlabel="Zaman (s)", ylabel="NIS",
    )

    final_time = truth[-1]["time_s"]

    for axis in axes[1:]:
        axis.axvspan(
            start_s,
            min(end_s, final_time),
            color="crimson",
            alpha=0.1,
            label="Enjekte edilen hata aralığı",
        )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    figure.savefig(output / "gps_fault.png", dpi=160)
    plt.close(figure)

    summary["all_passed"] = all(checks.values())

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"Sapma eklenen GPS ölçümü: {affected_count}")
    print(f"Sonuç klasörü: {output}")

    if not summary["all_passed"]:
        raise RuntimeError("GPS hata deneyinin kontrolleri başarısız.")


if __name__ == "__main__":
    main()