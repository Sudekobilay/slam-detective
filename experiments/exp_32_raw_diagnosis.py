"""EXP-32: Ham noktalarla hata hipotezlerinin kullanılabilirliği.

Hareket seçiciyi değiştirmez.
Her iki hipotez hesabı aynı ön tahmine bakar.
"""

import copy
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from estimation.online_diagnostic_agent import (
    OnlineDiagnosticAgent,
)
from estimation.raw_scan_hypotheses import (
    compare_raw_hypotheses,
    diagnose_raw,
)
from estimation.scan_alignment_guard import (
    ScanAlignmentGuard,
)
from storage.files import write_json, write_jsonl
from experiments.exp_30_scan_guard_audit import (
    fixed_recording,
    calibrate_healthy,
)
from storage.files import write_csv
from simulator.encoder import EncoderMeasurement
from simulator.imu import GyroMeasurement


def inspect_recording(events, normal, config):
    agent = OnlineDiagnosticAgent(
        strategy="passive",
        scan_guard=ScanAlignmentGuard(config),
    )

    rows = []

    for time, batch in groupby(
        events,
        key=lambda event: event["time_s"],
    ):
        batch = list(batch)

        packets = [
            event
            for event in batch
            if event["sensor_type"] == "lidar"
        ]

        if packets:
            # Aynı zamanın ön tahmini bağımsız kopyada
            # hesaplanır. Asıl kestirici değiştirilmez.
            preview = copy.deepcopy(agent.estimator)

            encoder = next(
                (
                    event
                    for event in batch
                    if event["sensor_type"] == "encoder"
                ),
                None,
            )
            gyro = next(
                (
                    event
                    for event in batch
                    if event["sensor_type"] == "gyro"
                ),
                None,
            )

            if encoder is not None:
                preview.predict(
                    EncoderMeasurement(
                        time_s=time,
                        **encoder["data"],
                    ),
                    GyroMeasurement(
                        time_s=time,
                        **gyro["data"],
                    ),
                )

            result = compare_raw_hypotheses(
                preview,
                packets,
                agent.mounts,
                config,
            )

        agent.process(batch)

        if packets:
            rows.append({
                "time_s": time,
                "old_status": (
                    agent.hypotheses[-1]["status"]
                ),
                "raw": result,
            })

    return rows


def main():
    root = Path(__file__).resolve().parents[1]

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_32_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    print(
        "EXP-32: ham nokta hipotezleri; "
        "hareket seçici değişmiyor.",
        flush=True,
    )

    config, guard_calibration, _ = calibrate_healthy()

    # Eşik yalnız sağlıklı kayıtlardan belirlenir.
    samples = []

    for normal in ("straight", "arc"):
        for seed in range(11, 21):
            events, _ = fixed_recording(
                normal,
                seed,
                0.0,
            )

            for row in inspect_recording(
                events,
                normal,
                config,
            ):
                if row["raw"]["status"] == "evaluated":
                    samples.append(
                        row["raw"]["models"][0][
                            "improvement_m2"
                        ]
                    )

    if not samples:
        raise RuntimeError(
            "Sağlıklı kalibrasyon desteği yok."
        )

    thresholds = {
        "minimum_improvement_m2": (
            max(samples) * 2.0 + 1e-6
        ),
        "minimum_model_gap_m2": 1e-6,
    }

    write_json(
        output / "calibration.json",
        {
            "guard": guard_calibration,
            "thresholds": thresholds,
            "healthy_samples": len(samples),
            "maximum_healthy_improvement_m2": (
                max(samples)
            ),
            "note": (
                "Mühendislik paylarıdır; "
                "istatistiksel güven düzeyi değildir."
            ),
        },
    )

    rows = []
    detail = []

    for normal in ("straight", "arc"):
        for seed in (401, 402, 403, 404, 405):
            for yaw in (0.0, 2.0, 5.0):
                events, _ = fixed_recording(
                    normal,
                    seed,
                    yaw,
                )

                for item in inspect_recording(
                    events,
                    normal,
                    config,
                ):
                    decision = diagnose_raw(
                        item["raw"],
                        thresholds,
                        config.max_alignment_rms_m,
                    )

                    detail.append({
                        "normal_motion": normal,
                        "seed": seed,
                        "fault_yaw_deg": yaw,
                        **item,
                        "decision": decision,
                    })

                    if item["time_s"] < 2.0 - 1e-9:
                        continue

                    rows.append({
                        "normal_motion": normal,
                        "seed": seed,
                        "fault_yaw_deg": yaw,
                        "time_s": item["time_s"],
                        "old_supported": (
                            item["old_status"]
                            == "evaluated"
                        ),
                        "raw_supported": (
                            item["raw"]["status"]
                            == "evaluated"
                        ),
                        **decision,
                    })

            print(
                f"{normal}/seed_{seed}: tamamlandı.",
                flush=True,
            )

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["normal_motion"],
            row["fault_yaw_deg"],
        )
        groups[key].append(row)

    summary = []

    for (normal, yaw), group in sorted(groups.items()):
        entry = {
            "normal_motion": normal,
            "fault_yaw_deg": yaw,
            "frames": len(group),
            "old_supported": sum(
                row["old_supported"]
                for row in group
            ),
            "raw_supported": sum(
                row["raw_supported"]
                for row in group
            ),
            "front_candidates": sum(
                row["hypothesis"] == "front_yaw"
                for row in group
            ),
            "other_fault_candidates": sum(
                row["hypothesis"] in (
                    "rear_yaw",
                    "heading_offset",
                )
                for row in group
            ),
            "nominal_consistent": sum(
                row["status"] == "nominal_consistent"
                for row in group
            ),
            "uncertain": sum(
                row["status"] == "uncertain"
                for row in group
            ),
        }

        summary.append(entry)

        print(
            f"{normal}/{yaw:g}deg: "
            f"eski destek={entry['old_supported']}/"
            f"{entry['frames']} | "
            f"ham destek={entry['raw_supported']}/"
            f"{entry['frames']} | "
            f"ön aday={entry['front_candidates']} | "
            f"belirsiz={entry['uncertain']}"
        )

    write_csv(output / "frames.csv", rows)
    write_csv(output / "group_summary.csv", summary)
    write_jsonl(
        output / "diagnostics.jsonl",
        detail,
    )

    write_json(
        output / "summary.json",
        {
            "experiment_id": "exp_32",
            "groups": summary,
            "thresholds": thresholds,
            "limitations": [
                "Tek karelik aday kararıdır; "
                "doğrulanmış sensör teşhisi değildir.",
                "Sadece ön LiDAR hata kayıtları "
                "değerlendirilir; diğer modelleri "
                "doğru ayırma henüz doğrulanmadı.",
                "Sabit kayıt, tek ortam, ideal LiDAR; "
                "GPS hipotezi yoktur.",
                "Puan geometrik kırpılmış MSE; "
                "posterior veya NIS değildir.",
                "Kalibrasyon 11–20; değerlendirme "
                "401–405. Bu seedler önceki "
                "deneylerde görüldü.",
                "Aralık [2,4]; alarm sonrası iki "
                "saniyelik pencere karşılaştırması "
                "değildir.",
                "Yeni hesap gölge izleyicidir; "
                "alarmı, haritayı ve hareket "
                "seçiciyi değiştirmez.",
            ],
        },
    )

    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()