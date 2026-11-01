"""Aynı sensör kaydında GPS'li ve GPS'siz EKF karşılaştırması."""

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

from estimation.pose_ekf import PoseEKF
from experiments.exp_11_replay_odometry import (
    evaluate,
    replay_measurements,
)
from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.robot import Pose


def snapshot(estimator, time_s):
    return {
        "time_s": time_s,
        "x_m": estimator.pose.x,
        "y_m": estimator.pose.y,
        "theta_rad": estimator.pose.theta,
        "estimated_gyro_bias_rad_s": float(estimator.state[3]),
        "position_std_proxy_m": math.sqrt(
            max(0.0, float(
                estimator.P[0, 0] + estimator.P[1, 1]
            ))
        ),
    }


def replay_ekf(input_path):
    """Gerçek rota veya simülasyon hata ayarlarını okumaz."""
    with input_path.open("r", encoding="utf-8") as file:
        events = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    estimators = {
        "ekf_no_gps": PoseEKF(initial_pose=Pose(1.0, 3.0, 0.0)),
        "ekf_gps": PoseEKF(initial_pose=Pose(1.0, 3.0, 0.0)),
    }
    paths = {name: [] for name in estimators}

    for time_s, group in itertools.groupby(
        events, key=lambda event: event["time_s"]
    ):
        group = list(group)

        encoder_events = [
            event for event in group
            if event["sensor_type"] == "encoder"
        ]
        gyro_events = [
            event for event in group
            if event["sensor_type"] == "gyro"
        ]

        if encoder_events or gyro_events:
            if len(encoder_events) != 1 or len(gyro_events) != 1:
                raise ValueError("Eksik veya yinelenen hareket paketi.")

            encoder = EncoderMeasurement(
                time_s=time_s, **encoder_events[0]["data"]
            )
            gyro = GyroMeasurement(
                time_s=time_s, **gyro_events[0]["data"]
            )

            for estimator in estimators.values():
                estimator.predict(encoder, gyro)

        for event in group:
            if event["sensor_type"] == "gps":
                estimators["ekf_gps"].update_gps(
                    GPSMeasurement(
                        time_s=time_s, **event["data"]
                    )
                )

        # Bu kayıt düzeninde her zaman grubu hareket saatiyle eşleşir.
        for name, estimator in estimators.items():
            if not math.isclose(
                time_s, estimator.time_s,
                rel_tol=0.0, abs_tol=1e-9,
            ):
                raise ValueError("Desteklenmeyen ara sensör zamanı.")

            paths[name].append(snapshot(estimator, time_s))

    return paths, estimators


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

    # Mevcut okuyucu sıralama, tekrar ve hareket aralıklarını kontrol eder.
    baseline_paths, _, update_count = replay_measurements(args.input)
    ekf_paths, estimators = replay_ekf(args.input)

    paths = {
        "encoder": baseline_paths["encoder"],
        "encoder_gyro": baseline_paths["encoder_gyro"],
        **ekf_paths,
    }

    # Gerçek rota yalnız tahminler tamamlandıktan sonra okunur.
    truth, evaluated = evaluate(paths, args.truth)

    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_12_{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    figure, axes = plt.subplots(
        1, 3, figsize=(17, 5), constrained_layout=True
    )

    summary = {
        "experiment": "EXP-12: EKF konum kestirimi",
        "run_id": run_id,
        "input_sha256": hashlib.sha256(
            args.input.read_bytes()
        ).hexdigest(),
        "truth_sha256": hashlib.sha256(
            args.truth.read_bytes()
        ).hexdigest(),
        "motion_intervals": update_count,
        "initial_pose_prior": {
            "x_m": 1.0, "y_m": 3.0, "theta_rad": 0.0,
        },
        "filter_parameters": {
            "state": ["x", "y", "theta", "gyro_bias"],
            "left_radius_m": 0.05,
            "right_radius_m": 0.05,
            "wheelbase_m": 0.30,
            "counts_per_revolution": 2048,
            "initial_covariance_diagonal": [
                1e-6, 1e-6, 0.005**2, 0.02**2
            ],
            "distance_noise_density": 0.005,
            "bias_random_walk": 0.001,
            "gps_variance_floor_m2": 1e-8,
        },
        "methods": {},
        "gps_updates": estimators["ekf_gps"].gps_updates,
        "gps_missing": estimators["ekf_gps"].gps_missing,
        "limitations": [
            "EKF konum kestirimi; SLAM değil.",
            "LiDAR henüz tahminde kullanılmıyor.",
            "Encoder dönüş bilgisi EKF'de kullanılmıyor.",
            "Süreç belirsizliği seçilmiş nominal modeldir; "
            "encoder kuantalamasının zaman korelasyonlarını "
            "tam temsil etmez.",
            "Belirsizlik kalibrasyonu veya bias'ın güvenilir "
            "kestirilebildiği kanıtlanmıyor.",
            "Bu sürüm ortak 10 Hz hareket saatine hizalı kayıt içindir.",
        ],
    }

    for name, rows in evaluated.items():
        write_csv(output / f"{name}_estimate.csv", paths[name])
        write_csv(output / f"{name}_evaluation.csv", rows)

        rmse = math.sqrt(
            sum(row["position_error_m"]**2 for row in rows)
            / len(rows)
        )
        summary["methods"][name] = {
            "position_rmse_m": rmse,
            "samples": len(rows),
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

        print(f"{name}: konum RMSE={rmse:.6f} m")

    for name, rows in ekf_paths.items():
        axes[2].plot(
            [row["time_s"] for row in rows],
            [row["position_std_proxy_m"] for row in rows],
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
        title="Nominal konum belirsizliği",
        xlabel="Zaman (s)",
        ylabel="sqrt(Pxx + Pyy) (m)",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)

    write_csv(
        output / "gps_innovations.csv",
        estimators["ekf_gps"].innovations,
    )

    checks = {
        "matching_sample_counts": all(
            len(path) == update_count + 1
            for path in paths.values()
        ),
        "finite_ekf_values": all(
            math.isfinite(value)
            for path in ekf_paths.values()
            for row in path
            for value in row.values()
        ),
    }

    summary["checks"] = checks
    summary["all_passed"] = all(checks.values())

    figure.savefig(output / "pose_ekf.png", dpi=160)
    plt.close(figure)

    with (output / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary, file, ensure_ascii=False,
            indent=2, allow_nan=False,
        )

    print(f"GPS güncellemesi: {summary['gps_updates']}")
    print(f"Eksik GPS: {summary['gps_missing']}")
    print(
        "Kayıt kontrolleri: "
        + ("PASS" if summary["all_passed"] else "FAIL")
    )
    print(f"Sonuç klasörü: {output}")

    if not summary["all_passed"]:
        raise RuntimeError("EKF kayıt kontrolleri başarısız.")


if __name__ == "__main__":
    main()
    