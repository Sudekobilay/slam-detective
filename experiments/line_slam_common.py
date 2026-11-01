import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from estimation.gps_gate import GatedLineEKFSLAM, GatedPoseEKF

import matplotlib.pyplot as plt

from estimation.line_ekf_slam import LineEKFSLAM
from estimation.pose_ekf import PoseEKF
from estimation.lidar_update_hold import LidarUpdateHold
from estimation.mounting_hypotheses import compare_mounting_hypotheses
from perception.line_association import (
    associate_lines,
    transform_line,
)
from perception.line_features import extract_lines
from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.mounting import SensorMount
from simulator.robot import Pose, wrap_angle


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(estimator, time_s):
    return {
        "time_s": time_s,
        "x_m": estimator.pose.x,
        "y_m": estimator.pose.y,
        "theta_rad": estimator.pose.theta,
    }


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(
    input_path, use_gps, gps_gate=False, exclude_lidar=None,
    lidar_hold_nis=None,
):
    # Bu fonksiyon gerçek konum dosyasını okumaz.
    events = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not events:
        raise ValueError("Ölçüm kaydı boş.")

    previous_time = -math.inf
    seen = set()

    for event in events:
        time_s = float(event["time_s"])

        if (
            event["schema_version"] != 1
            or not math.isfinite(time_s)
            or time_s < 0
            or time_s < previous_time
        ):
            raise ValueError("Geçersiz şema veya zaman sırası.")

        key = (
            event["sensor_type"],
            event["sensor_id"],
            time_s,
        )
        if key in seen:
            raise ValueError(f"Tekrarlanan ölçüm: {key}")

        seen.add(key)
        previous_time = time_s

    # Algoritmanın bildiği başlangıç ve nominal montaj bilgileri.
    initial_pose = Pose(1.0, 3.0, 0.0)
    mounts = {
        "front": SensorMount(x=0.25),
        "rear": SensorMount(x=-0.25, yaw=math.pi),
    }

    baseline_type = GatedPoseEKF if gps_gate else PoseEKF
    slam_type = GatedLineEKFSLAM if gps_gate else LineEKFSLAM

    baseline = baseline_type(initial_pose=initial_pose)
    slam = slam_type(initial_pose=initial_pose)

    paths = {"baseline": [], "line_slam": []}
    associations = []
    map_history = []
    counts = Counter()
    slam.lidar_update_hold = (
        LidarUpdateHold(lidar_hold_nis) if lidar_hold_nis is not None else None
    )
    slam.mounting_hypotheses = []
    slam.lidar_innovations = []
    slam.lidar_scan_diagnostics = []

    for time_s, grouped in groupby(
        events, key=lambda event: float(event["time_s"])
    ):
        packets = list(grouped)

        encoders = [
            event for event in packets
            if event["sensor_type"] == "encoder"
        ]
        gyros = [
            event for event in packets
            if event["sensor_type"] == "gyro"
        ]

        if encoders or gyros:
            if len(encoders) != 1 or len(gyros) != 1:
                raise ValueError("Encoder ve gyro eşleştirilemedi.")

            encoder = EncoderMeasurement(
                time_s=time_s, **encoders[0]["data"]
            )
            gyro = GyroMeasurement(
                time_s=time_s, **gyros[0]["data"]
            )

            baseline.predict(encoder, gyro)
            slam.predict(encoder, gyro)
            counts["prediction_intervals"] += 1

        if not math.isclose(
            slam.time_s, time_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "Ölçüm zamanı kestiricinin zamanı ile uyuşmuyor."
            )

        if use_gps:
            for event in packets:
                if event["sensor_type"] == "gps":
                    measurement = GPSMeasurement(
                        time_s=time_s, **event["data"]
                    )
                    baseline.update_gps(measurement)
                    slam.update_gps(measurement)

        # Aynı andaki LiDAR'lar için işlem sırası sabittir.
        lidar_packets = sorted(
            (
                event for event in packets
                if (
                    event["sensor_type"] == "lidar"
                    and event["sensor_id"] != exclude_lidar
                )
            ),
            key=lambda event: event["sensor_id"],
        )

        # İki sensörün tanısal karşılaştırması aynı ortak tahmini kullanır.
        # GPS bu anda işlenmiştir; hiçbir LiDAR henüz güncellenmemiştir.
        diagnostic_references = slam.map_features()
        feature_cache = {}
        hypothesis_observations = []
        hold_evidence = []

        for event in lidar_packets:
            sensor_id = event["sensor_id"]
            if sensor_id not in mounts:
                raise ValueError(
                    f"Nominal montajı tanımsız sensör: {sensor_id}"
                )
            mount = mounts[sensor_id]
            features = extract_lines(event["data"]["scan"]["points"])
            feature_cache[sensor_id] = features
            world_features = [
                transform_line(feature, mount.world_pose(slam.pose))
                for feature in features
            ]
            diagnostic_matches = associate_lines(
                world_features, diagnostic_references
            )
            statuses = Counter()
            matched_innovations = []

            for match in diagnostic_matches:
                status = match["status"]
                statuses[status] += 1
                innovation = None
                if status == "matched":
                    innovation = slam.line_innovation(
                        match["reference_index"],
                        features[match["observation_index"]],
                        mount,
                    )
                    matched_innovations.append(innovation)
                    hold_evidence.append({
                        "sensor_id": sensor_id,
                        "observation_index": match["observation_index"],
                        "nis": innovation["nis"],
                    })
                    hypothesis_observations.append((
                        sensor_id,
                        match["reference_index"],
                        features[match["observation_index"]],
                        mount,
                    ))

                slam.lidar_innovations.append({
                    "time_s": time_s,
                    "sensor_id": sensor_id,
                    "observation_index": match["observation_index"],
                    "status": status,
                    "reference_index": match["reference_index"],
                    "innovation": innovation,
                    "prior_stage": "after_gps_before_all_lidar",
                })

            n_matched = len(matched_innovations)

            def matched_mean(field, absolute=False):
                if not n_matched:
                    return None
                values = [item[field] for item in matched_innovations]
                if absolute:
                    values = [abs(value) for value in values]
                return sum(values) / n_matched

            slam.lidar_scan_diagnostics.append({
                "time_s": time_s,
                "sensor_id": sensor_id,
                "prior_landmark_count": slam.landmark_count,
                "feature_count": len(features),
                "matched": statuses["matched"],
                "no_candidate": statuses["no_candidate"],
                "ambiguous": statuses["ambiguous"],
                "conflict": statuses["conflict"],
                "mean_nis": matched_mean("nis"),
                "max_nis": (
                    max(item["nis"] for item in matched_innovations)
                    if n_matched else None
                ),
                "mean_abs_rho_residual_m": matched_mean(
                    "residual_rho_m", absolute=True
                ),
                "mean_abs_alpha_residual_rad": matched_mean(
                    "residual_alpha_rad", absolute=True
                ),
            })

        if lidar_packets:
            comparison = compare_mounting_hypotheses(
                slam, hypothesis_observations
            )
            slam.mounting_hypotheses.append({
                "time_s": time_s,
                "prior_stage": "after_gps_before_all_lidar",
                **comparison,
            })

        if lidar_packets and slam.lidar_update_hold is not None:
            slam.lidar_update_hold.observe(time_s, hold_evidence)

        for event in lidar_packets:
            sensor_id = event["sensor_id"]

            if sensor_id not in mounts:
                raise ValueError(
                    f"Nominal montajı tanımsız sensör: {sensor_id}"
                )

            mount = mounts[sensor_id]
            features = feature_cache[sensor_id]

            if (
                slam.lidar_update_hold is not None
                and slam.lidar_update_hold.active
            ):
                counts["lidar_scans"] += 1
                counts["extracted_features"] += len(features)
                counts["held_scans"] += 1
                counts["held_features"] += len(features)
                for observation_index in range(len(features)):
                    associations.append({
                        "time_s": time_s,
                        "sensor_id": sensor_id,
                        "observation_index": observation_index,
                        "status": "held",
                        "landmark_id": None,
                        "nis": None,
                    })
                continue

            sensor_pose = mount.world_pose(slam.pose)
            world_observations = [
                transform_line(feature, sensor_pose)
                for feature in features
            ]

            # Eşleştirme, bu taramanın güncellemelerinden önce yapılır.
            references = slam.map_features()
            matches = associate_lines(
                world_observations, references
            )

            for match in matches:
                feature = features[match["observation_index"]]
                status = match["status"]
                landmark_id = None
                nis = None

                if status == "matched":
                    landmark_id = match["reference_index"]
                    nis = float(
                        slam.update_line(
                            landmark_id, feature, mount
                        )
                    )
                elif status == "no_candidate":
                    # İlk gözlem yalnızca haritaya eklenir.
                    landmark_id = slam.add_line(feature, mount)

                # Belirsiz ve çakışan eşleştirmeler kullanılmaz.
                counts[status] += 1
                associations.append({
                    "time_s": time_s,
                    "sensor_id": sensor_id,
                    "observation_index": match[
                        "observation_index"
                    ],
                    "status": status,
                    "landmark_id": landmark_id,
                    "nis": nis,
                })

            counts["lidar_scans"] += 1
            counts["extracted_features"] += len(features)

        paths["baseline"].append(snapshot(baseline, time_s))
        paths["line_slam"].append(snapshot(slam, time_s))
        map_history.append({
            "time_s": time_s,
            "landmark_count": slam.landmark_count,
            "line_updates": slam.line_updates,
            "lidar_hold_active": (
                slam.lidar_update_hold.active
                if slam.lidar_update_hold is not None else False
            ),
        })

    return paths, slam, associations, map_history, counts


def evaluate(paths, truth_path):
    # Gerçek konum ilk kez burada, kestirim bittikten sonra okunur.
    with truth_path.open(newline="", encoding="utf-8") as handle:
        truth_rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]

    truth_by_time = {
        round(row["time_s"], 9): row for row in truth_rows
    }

    evaluated = {}
    metrics = {}

    for method, rows in paths.items():
        evaluated[method] = []

        for row in rows:
            truth = truth_by_time[round(row["time_s"], 9)]

            position_error = math.hypot(
                row["x_m"] - truth["x_m"],
                row["y_m"] - truth["y_m"],
            )
            heading_error = wrap_angle(
                row["theta_rad"] - truth["theta_rad"]
            )

            evaluated[method].append({
                **row,
                "position_error_m": position_error,
                "heading_error_rad": heading_error,
            })

        samples = evaluated[method]
        metrics[method] = {
            "position_rmse_m": math.sqrt(
                sum(row["position_error_m"] ** 2 for row in samples)
                / len(samples)
            ),
            "heading_rmse_rad": math.sqrt(
                sum(row["heading_error_rad"] ** 2 for row in samples)
                / len(samples)
            ),
        }

    return truth_rows, evaluated, metrics


def main(*, experiment_id="exp_18", description=None, defaults=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--gps", action="store_true")
    parser.add_argument("--gps-gate", action="store_true")
    parser.add_argument(
        "--exclude-lidar",
        choices=["front", "rear"],
        default=None,
    )
    parser.add_argument(
        "--lidar-hold-nis", type=float, default=None,
        help="Bireysel LiDAR NIS eşiği; aşılırsa LiDAR güncellemeleri kilitlenir.",
    )
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()

    if args.lidar_hold_nis is not None and (
        not math.isfinite(args.lidar_hold_nis) or args.lidar_hold_nis <= 0
    ):
        parser.error("--lidar-hold-nis pozitif ve sonlu olmalı.")

    if args.gps_gate and not args.gps:
        parser.error("--gps-gate kullanmak için --gps de gerekli.")

    print(f"Deney: {experiment_id} — {description or 'Çizgi EKF-SLAM'}")

    original_hashes = {
        "input": file_hash(args.input),
        "truth": file_hash(args.truth),
    }

    paths, slam, associations, map_history, counts = replay(
        args.input, args.gps, args.gps_gate, args.exclude_lidar, args.lidar_hold_nis
    )

    truth, evaluated, metrics = evaluate(paths, args.truth)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = Path("results") / f"{experiment_id}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    for method, rows in evaluated.items():
        write_csv(output / f"{method}.csv", rows)

    write_csv(output / "map_history.csv", map_history)

    with (output / "associations.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in associations:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    write_csv(
        output / "lidar_scan_diagnostics.csv",
        slam.lidar_scan_diagnostics,
    )
    with (output / "lidar_innovations.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in slam.lidar_innovations:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    diagnostic_summary = {}
    for sensor_id in sorted({
        row["sensor_id"] for row in slam.lidar_scan_diagnostics
    }):
        scans = [
            row for row in slam.lidar_scan_diagnostics
            if row["sensor_id"] == sensor_id
        ]
        nis_values = [
            row["innovation"]["nis"]
            for row in slam.lidar_innovations
            if row["sensor_id"] == sensor_id
            and row["innovation"] is not None
        ]
        diagnostic_summary[sensor_id] = {
            "scan_count": len(scans),
            "feature_count": sum(row["feature_count"] for row in scans),
            "matched": sum(row["matched"] for row in scans),
            "no_candidate": sum(row["no_candidate"] for row in scans),
            "ambiguous": sum(row["ambiguous"] for row in scans),
            "conflict": sum(row["conflict"] for row in scans),
            "mean_nis": (
                sum(nis_values) / len(nis_values) if nis_values else None
            ),
            "max_nis": max(nis_values) if nis_values else None,
        }

    with (output / "mounting_hypotheses.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in slam.mounting_hypotheses:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    hypothesis_rows = []
    for row in slam.mounting_hypotheses:
        for model in row["models"]:
            hypothesis_rows.append({
                "time_s": row["time_s"],
                "front_matches": row["matched_counts"]["front"],
                "rear_matches": row["matched_counts"]["rear"],
                "nominal_nll": row["nominal_nll"],
                **model,
            })
    write_csv(output / "mounting_hypotheses.csv", hypothesis_rows)

    map_lines = slam.map_features()
    map_data = [
        {
            "landmark_id": index,
            "rho_m": float(slam.state[4 + 2 * index]),
            "alpha_rad": float(slam.state[5 + 2 * index]),
            "start": list(feature.start),
            "end": list(feature.end),
        }
        for index, feature in enumerate(map_lines)
    ]

    (output / "estimated_map.json").write_text(
        json.dumps(map_data, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    (output / "joint_estimate.json").write_text(
        json.dumps(
            {
                "state": slam.state.tolist(),
                "covariance": slam.P.tolist(),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    checks = {
        "map_created": slam.landmark_count > 0,
        "line_updates_performed": slam.line_updates > 0,
        "state_dimension": (
            len(slam.state) == 4 + 2 * slam.landmark_count
        ),
        "source_files_unchanged": (
            original_hashes["input"] == file_hash(args.input)
            and original_hashes["truth"] == file_hash(args.truth)
        ),
    }

    hold = slam.lidar_update_hold
    hold_summary = {
        "enabled": hold is not None,
        "threshold": hold.nis_threshold if hold is not None else None,
        "active": hold.active if hold is not None else False,
        "trigger_time_s": hold.trigger_time_s if hold is not None else None,
        "trigger_evidence": hold.trigger_evidence if hold is not None else [],
        "held_scans": counts["held_scans"],
    }

    summary = {
        "experiment_id": experiment_id,
        "experiment_description": description,
        "gps_enabled": args.gps,
        "lidar_update_hold": hold_summary,
        "gps_gate_enabled": args.gps_gate,
        "slam_gps_rejected": getattr(slam, "gps_rejected", 0),
        "slam_gps_gate_log": getattr(slam, "gps_gate_log", []),
        "input_sha256": original_hashes["input"],
        "truth_sha256": original_hashes["truth"],
        "metrics": metrics,
        "counts": dict(counts),
        "landmark_count": slam.landmark_count,
        "line_updates": slam.line_updates,
        "checks": checks,
        "lidar_diagnostics": diagnostic_summary,
        "diagnostic_prior_stage": "after_gps_before_all_lidar",
        "mounting_hypothesis_models": [
            "front_yaw", "rear_yaw", "heading_offset"
        ],
        "mounting_hypothesis_search_deg": {"limit": 10.0, "step": 0.5},
        "limitations": [
            "Çizgi ölçüm kovaryansı nominal ve sabittir.",
            "Özellikler arası korelasyon modellenmemiştir.",
            "Eşleştirme geometrik eşiklere dayanır.",
            "Yeni çizgiler arasında birleştirme uygulanmaz.",
            "Çizgi uçları bağımsız EKF durumları değildir.",
            "LiDAR kilidi isteğe bağlı ve deney sonuna kadar kalıcıdır.",
            "Kilit açıkken tüm LiDAR düzeltmeleri ve çizgi başlatmaları askıya alınır.",
            "Kilit bir hata kaynağı teşhisi değildir; genel yanlış alarm oranı kalibre edilmemiştir.",
            "GPS kapısı, hata kaynağını kesin olarak teşhis etmez.",
            "Tanısal eşleştirmeler ortak LiDAR öncesi tahmini kullanır.",
            "Tanısal ve sıralı güncelleme eşleştirmeleri farklı olabilir.",
            "NIS yalnız eşleştirilmiş çizgilerde hesaplanır.",
            "Eşleştirme seçimi ve ortak harita, NIS yorumunu etkiler.",
            "Ortalama NIS, bağımsız bir ki-kare testi değildir.",
            "Bu kayıtlar otomatik hata kaynağı teşhisi yapmaz.",
            "Hipotezler aynı nominal eşleşmeleri kullanır; seçim yanlılığı vardır.",
            "Hipotezlerde ölçüm korelasyonu ortak kovaryansla taşınır; özellik çıkarımı korelasyonları modellenmez.",
            "Hipotez skoru bir posterior olasılık değildir; parametre araması uyumu artırabilir.",
            "GPS sapması henüz hipotez bankasında modellenmez.",
        ],
        "excluded_lidar": args.exclude_lidar,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        [row["x_m"] for row in truth],
        [row["y_m"] for row in truth],
        "k--",
        label="Gerçek rota",
    )

    for method, rows in evaluated.items():
        axes[0].plot(
            [row["x_m"] for row in rows],
            [row["y_m"] for row in rows],
            label=method,
        )
        axes[1].plot(
            [row["time_s"] for row in rows],
            [row["position_error_m"] for row in rows],
            label=method,
        )

    for feature in map_lines:
        axes[0].plot(
            [feature.start[0], feature.end[0]],
            [feature.start[1], feature.end[1]],
            color="gray",
            alpha=0.6,
        )

    axes[0].set(
        title="Kestirilen rota ve çizgi haritası",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(
        title="Konum hatası",
        xlabel="Zaman (s)",
        ylabel="Hata (m)",
    )

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)

    for method, values in metrics.items():
        print(
            f"{method}: "
            f"konum RMSE={values['position_rmse_m']:.6f} m | "
            f"yön RMSE={values['heading_rmse_rad']:.6f} rad"
        )

    if args.gps_gate:
        print(f"SLAM tarafından reddedilen GPS: {slam.gps_rejected}")

    if hold is not None:
        print(f"LiDAR güncelleme kilidi: {'AKTİF' if hold.active else 'tetiklenmedi'}")
        print(f"Kilit başlangıcı: {hold.trigger_time_s}")
        print(f"Askıya alınan tarama: {counts['held_scans']}")

    print(f"Haritadaki çizgi sayısı: {slam.landmark_count}")
    print(f"Çizgi güncellemesi: {slam.line_updates}")
    print(f"Eşleştirme durumları: {dict(counts)}")

    for sensor_id, values in diagnostic_summary.items():
        mean_nis = values["mean_nis"]
        nis_text = f"{mean_nis:.6f}" if mean_nis is not None else "yok"
        print(
            f"Tanısal/{sensor_id}: eşleşen={values['matched']} | "
            f"ortalama NIS={nis_text}"
        )

    for row in slam.mounting_hypotheses:
        if row["status"] != "evaluated":
            continue
        details = " | ".join(
            f"{model['hypothesis']}: açı={model['best_offset_deg']:+.1f} derece, "
            f"iyileşme={model['nll_improvement_over_nominal']:.3f}"
            for model in row["models"]
        )
        print(f"Hipotez/t={row['time_s']:.1f}: {details}")

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"Sonuç klasörü: {output.resolve()}")