"""EXP-34: Üç geometrik hata kaynağının doğrulanması."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from estimation.raw_scan_hypotheses import diagnose_raw
from estimation.scan_alignment_guard import ScanGuardConfig
from experiments.diagnosis_validation_common import (
    fault_recording,
    learn_reference,
    inspect_frozen,
    confirm_sequence,
)
from experiments.exp_24_motion_comparison import (
    write_json,
    write_jsonl,
)
from experiments.line_slam_common import write_csv


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = root / "results" / f"exp_34_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    print(
        "EXP-34: üç hata kaynağı; "
        "t>=2 sabit öğrenilmiş referans; "
        "alarm testi değildir.",
        flush=True,
    )

    config = ScanGuardConfig(1.0, 0.02)
    healthy = []

    # Kalibrasyon yalnız sağlıklı kayıtlardan yapılır.
    for normal in ("straight", "arc"):
        for seed in range(11, 21):
            events, _ = fault_recording(
                normal,
                seed,
                "healthy",
                0.0,
            )

            healthy.extend(
                inspect_frozen(
                    events,
                    learn_reference(events),
                    config,
                )
            )

    supported = [
        row["raw"]
        for row in healthy
        if row["raw"]["status"] == "evaluated"
    ]

    if not supported:
        raise RuntimeError(
            "Kalibrasyon desteği yok."
        )

    thresholds = {
        "minimum_improvement_m2": (
            2.0 * max(
                row["models"][0]["improvement_m2"]
                for row in supported
            ) + 1e-6
        ),
        "minimum_model_gap_m2": 1e-6,
    }

    max_rms = max(
        row["models"][0]["max_sensor_rms_m"]
        for row in supported
    ) + 0.01

    write_json(
        output / "calibration.json",
        {
            "seeds": list(range(11, 21)),
            "healthy_sequences": 20,
            "supported_frames": len(supported),
            "thresholds": thresholds,
            "max_rms_m": max_rms,
            "note": (
                "Yalnız sağlıklı kalibrasyon; "
                "mühendislik payları güven "
                "olasılığı değildir."
            ),
        },
    )

    cases = [("healthy", 0.0)] + [
        (source, angle)
        for source in (
            "front_yaw",
            "rear_yaw",
            "heading_offset",
        )
        for angle in (-5.0, -2.0, 2.0, 5.0)
    ]

    rows = []

    for normal in ("straight", "arc"):
        for seed in (601, 602, 603, 604, 605):
            nominal, records = fault_recording(
                normal,
                seed,
                "healthy",
                0.0,
            )

            reference = learn_reference(nominal)

            for source, angle in cases:
                events, case_records = fault_recording(
                    normal,
                    seed,
                    source,
                    angle,
                )

                target = (
                    source.removesuffix("_yaw")
                    if source in (
                        "front_yaw",
                        "rear_yaw",
                    )
                    else None
                )

                isolation = (
                    records["truth"]
                    == case_records["truth"]
                    and len(events) == len(nominal)
                    and all(
                        first == second
                        for first, second
                        in zip(events, nominal)
                        if (
                            target is None
                            or first["sensor_id"] != target
                            or first["time_s"] < 2.0
                        )
                    )
                )

                if not isolation:
                    raise RuntimeError(
                        "Hata izolasyonu veya "
                        "gerçek rota değişti."
                    )

                frames = inspect_frozen(
                    events,
                    reference,
                    config,
                    heading_perturbation_deg=(
                        angle
                        if source == "heading_offset"
                        else 0.0
                    ),
                )

                for frame in frames:
                    frame["decision"] = diagnose_raw(
                        frame["raw"],
                        thresholds,
                        max_rms,
                    )

                result = confirm_sequence(frames)

                if result["status"] == "confirmed":
                    if source == "healthy":
                        outcome = "false_fault"
                    elif result["hypothesis"] == source:
                        outcome = "correct"
                    else:
                        outcome = "wrong_source"

                elif (
                    result["status"]
                    == "nominal_consistent"
                ):
                    outcome = (
                        "correct"
                        if source == "healthy"
                        else "missed_fault"
                    )

                else:
                    outcome = "uncertain"

                row = {
                    "normal_motion": normal,
                    "seed": seed,
                    "source": source,
                    "angle_deg": angle,
                    **result,
                    "outcome": outcome,
                    "frames": len(frames),
                    "supported_frames": sum(
                        frame["raw"]["status"]
                        == "evaluated"
                        for frame in frames
                    ),
                    "fault_isolation_passed": isolation,
                }

                rows.append(row)

                folder = (
                    output
                    / f"normal_{normal}"
                    / f"seed_{seed}"
                    / f"{source}_{angle:g}"
                )
                folder.mkdir(parents=True)

                write_jsonl(
                    folder / "measurements.jsonl",
                    events,
                )
                write_jsonl(
                    folder / "diagnostics.jsonl",
                    frames,
                )
                write_csv(
                    folder / "ground_truth.csv",
                    case_records["truth"],
                )
                write_json(
                    folder / "summary.json",
                    row,
                )

            print(
                f"{normal}/seed_{seed}: "
                f"{len(rows)} kayıt tamamlandı.",
                flush=True,
            )

    groups = defaultdict(list)

    for row in rows:
        groups[
            row["normal_motion"],
            row["source"],
            row["angle_deg"],
        ].append(row)

    summary = []

    for (normal, source, angle), group in sorted(
        groups.items()
    ):
        counts = Counter(
            row["outcome"]
            for row in group
        )

        entry = {
            "normal_motion": normal,
            "source": source,
            "angle_deg": angle,
            "trials": len(group),
            **{
                key: counts[key]
                for key in (
                    "correct",
                    "wrong_source",
                    "missed_fault",
                    "false_fault",
                    "uncertain",
                )
            },
        }

        summary.append(entry)

        print(
            f"{normal}/{source}/{angle:+g}: "
            f"doğru={counts['correct']}/5 | "
            f"yanlış kaynak={counts['wrong_source']} | "
            f"kaçan={counts['missed_fault']} | "
            f"yanlış hata={counts['false_fault']} | "
            f"belirsiz={counts['uncertain']}"
        )

    write_csv(output / "trials.csv", rows)
    write_csv(output / "group_summary.csv", summary)

    write_json(
        output / "summary.json",
        {
            "experiment_id": "exp_34",
            "trial_count": len(rows),
            "groups": summary,
            "all_isolation_checks_passed": all(
                row["fault_isolation_passed"]
                for row in rows
            ),
            "limitations": [
                "Bu sabit referanslı teşhis "
                "doğrulamasıdır; çevrim içi alarm "
                "başarımı veya hareket seçimi "
                "karşılaştırması değildir.",
                "Harita her koşulda önceden "
                "belirlenen t=2 anında sabitlenir; "
                "hata etiketi sabitleme kararında "
                "kullanılmaz.",
                "Yön hatası karşılaştırma kopyasına "
                "uygulanan açı ofsetidir; gyro bias "
                "veya konum sürüklenmesi deneyi değildir.",
                "İlk iki ardışık aynı model/işaret "
                "kararı saklanır; istatistiksel "
                "kesinlik veya otomatik toparlanma "
                "iddiası yoktur.",
                "Yeni seedler 601–605; tek ortam "
                "ve ideal LiDAR. İki hata eşzamanlı "
                "uygulanmaz.",
                "GPS hipotezi henüz yoktur; yalnız "
                "üç geometrik model değerlendirilir.",
            ],
        },
    )

    print("Hata izolasyonu kontrolleri: PASS")
    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()