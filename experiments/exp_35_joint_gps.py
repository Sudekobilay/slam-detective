"""EXP-35: Çoklu LiDAR, gyro, encoder ve GPS değerlendirmesi."""

import copy
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from estimation.gps_consistency import (
    GPSConsistencyMonitor,
    joint_status,
)
from estimation.online_diagnostic_agent import (
    OnlineDiagnosticAgent,
)
from estimation.scan_alignment_guard import (
    ScanAlignmentGuard,
)
from experiments.diagnosis_validation_common import (
    fault_recording,
)
from storage.files import write_json, write_jsonl
from experiments.exp_30_scan_guard_audit import (
    calibrate_healthy,
)
from evaluation.trajectory import evaluate
from storage.files import write_csv
from simulator.gps import GPSMeasurement


CASES = (
    ("healthy", "healthy", 0.0, 0.0, 0.0),
    ("gps_positive", "healthy", 0.0, 0.5, 0.3),
    ("gps_negative", "healthy", 0.0, -0.5, -0.3),
    ("front_fault", "front_yaw", 5.0, 0.0, 0.0),
    ("rear_fault", "rear_yaw", 5.0, 0.0, 0.0),
    ("front_and_gps", "front_yaw", 5.0, 0.5, 0.3),
)


def add_gps_bias(events, dx, dy):
    result = copy.deepcopy(events)

    for event in result:
        if (
            event["sensor_type"] == "gps"
            and event["time_s"] >= 2.0
            and event["data"]["available"]
        ):
            event["data"]["x_m"] += dx
            event["data"]["y_m"] += dy

    return result


def replay_joint(events, config):
    def agent():
        return OnlineDiagnosticAgent(
            strategy="passive",
            scan_guard=ScanAlignmentGuard(config),
        )

    # Bu kestiriciye GPS düzeltmesi verilmez.
    reference = agent()

    filters = {
        name: agent()
        for name in (
            "no_gps",
            "plain_gps",
            "guarded_gps",
        )
    }

    monitor = GPSConsistencyMonitor()
    accepted = {
        name: 0
        for name in filters
    }
    rejected = 0

    for _, batch in groupby(
        events,
        key=lambda event: event["time_s"],
    ):
        batch = list(batch)

        reference.process(batch)

        for item in filters.values():
            item.process(batch)

        for event in batch:
            if event["sensor_type"] != "gps":
                continue

            packet = GPSMeasurement(
                time_s=event["time_s"],
                **event["data"],
            )

            check = monitor.observe(
                reference.estimator,
                packet,
            )

            for name in (
                "plain_gps",
                "guarded_gps",
            ):
                reject = (
                    name == "guarded_gps"
                    and packet.available
                    and (
                        check["instant_reject"]
                        or monitor.confirmed
                    )
                )

                if reject:
                    rejected += 1
                else:
                    accepted[name] += int(
                        filters[
                            name
                        ].estimator.update_gps(packet)
                    )

    status = joint_status(
        reference.hold.active,
        monitor.confirmed,
    )

    return (
        reference,
        filters,
        monitor,
        accepted,
        rejected,
        status,
    )


def main(argv=None):
    root = Path(__file__).resolve().parents[1]

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = root / "results" / f"exp_35_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    print(
        "EXP-35: ortak kayıtta GPS füzyonu ve "
        "bağımsız referansla tutarsızlık izleme.",
        flush=True,
    )

    config, calibration, _ = calibrate_healthy()

    write_json(
        output / "calibration.json",
        calibration,
    )

    rows = []

    for normal in ("straight", "arc"):
        for seed in (701, 702, 703, 704, 705):
            for case, source, angle, dx, dy in CASES:
                events, records = fault_recording(
                    normal,
                    seed,
                    source,
                    angle,
                )

                biased = add_gps_bias(
                    events,
                    dx,
                    dy,
                )

                isolation = all(
                    first == second
                    for first, second
                    in zip(events, biased)
                    if (
                        first["sensor_type"] != "gps"
                        or first["time_s"] < 2.0
                        or not first["data"]["available"]
                    )
                )

                if not isolation:
                    raise RuntimeError(
                        "GPS sapması başka ölçümleri değiştirdi."
                    )

                (
                    reference,
                    filters,
                    monitor,
                    accepted,
                    rejected,
                    status,
                ) = replay_joint(biased, config)

                folder = (
                    output
                    / f"normal_{normal}"
                    / f"seed_{seed}"
                    / case
                )
                folder.mkdir(parents=True)

                write_jsonl(
                    folder / "measurements.jsonl",
                    biased,
                )

                truth = folder / "ground_truth.csv"

                write_csv(
                    truth,
                    records["truth"],
                )

                paths = {
                    name: item.path
                    for name, item in filters.items()
                }

                _, evaluated, metrics = evaluate(
                    paths,
                    truth,
                )

                for name, path in evaluated.items():
                    write_csv(
                        folder / f"{name}_evaluation.csv",
                        path,
                    )

                write_csv(
                    folder / "gps_diagnostics.csv",
                    monitor.history,
                )

                row = {
                    "normal_motion": normal,
                    "seed": seed,
                    "case": case,
                    "joint_status": status,
                    "geometric_alarm": (
                        reference.hold.active
                    ),
                    "gps_confirmed": monitor.confirmed,
                    "gps_confirmation_time_s": (
                        monitor.confirmation_time_s
                    ),
                    "gps_rejected": rejected,
                    "plain_gps_updates": (
                        accepted["plain_gps"]
                    ),
                    "guarded_gps_updates": (
                        accepted["guarded_gps"]
                    ),
                    "no_gps_rmse_m": metrics[
                        "no_gps"
                    ]["position_rmse_m"],
                    "plain_gps_rmse_m": metrics[
                        "plain_gps"
                    ]["position_rmse_m"],
                    "guarded_gps_rmse_m": metrics[
                        "guarded_gps"
                    ]["position_rmse_m"],
                    "gps_isolation_passed": isolation,
                }

                rows.append(row)

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
            row["case"],
        ].append(row)

    summary = []

    for (normal, case), trials in sorted(
        groups.items()
    ):
        entry = {
            "normal_motion": normal,
            "case": case,
            "trials": len(trials),
            "geometric_alarm_trials": sum(
                row["geometric_alarm"]
                for row in trials
            ),
            "gps_confirmed_trials": sum(
                row["gps_confirmed"]
                for row in trials
            ),
            **{
                f"mean_{key}": (
                    sum(row[key] for row in trials)
                    / len(trials)
                )
                for key in (
                    "no_gps_rmse_m",
                    "plain_gps_rmse_m",
                    "guarded_gps_rmse_m",
                )
            },
        }

        summary.append(entry)

        print(
            f"{normal}/{case}: "
            f"geometri={entry['geometric_alarm_trials']}/5 | "
            f"GPS={entry['gps_confirmed_trials']}/5 | "
            "RMSE yok/korumasız/korumalı="
            f"{entry['mean_no_gps_rmse_m']:.6f}/"
            f"{entry['mean_plain_gps_rmse_m']:.6f}/"
            f"{entry['mean_guarded_gps_rmse_m']:.6f} m"
        )

    write_csv(output / "trials.csv", rows)
    write_csv(output / "group_summary.csv", summary)

    write_json(
        output / "summary.json",
        {
            "experiment_id": "exp_35",
            "groups": summary,
            "trial_count": len(rows),
            "all_gps_isolation_passed": all(
                row["gps_isolation_passed"]
                for row in rows
            ),
            "limitations": [
                "Sabit kayıt değerlendirmesidir; "
                "bu deney GPS alarmıyla yeni "
                "hareket başlatmaz.",
                "GPS füzyonlu kestirim ile GPS "
                "almayan teşhis referansı ayrı "
                "tutulur; GPS kendi doğrulama "
                "referansını değiştirmez.",
                "GPS-referans uyuşmazlığı kesin "
                "GPS arızası teşhisi değildir; "
                "ortak küresel konum kayması "
                "ayırt edilemeyebilir.",
                "Başlangıç pozu nominal küresel "
                "çerçevede verilir; GPS referans "
                "noktası robot merkezi kabul edilir.",
                "GPS 1–3 saniye arasında yoktur. "
                "t=2 sapması GPS dönmeden "
                "gözlenemez; üç ardışık aşım gerekir.",
                "NIS kapısı tek ölçümde reddeder; "
                "üç aşım kaynak şüphesini kalıcı "
                "işaretler. Otomatik toparlanma yoktur.",
                "Tek ortam, ideal LiDAR, nominal "
                "kovaryanslar; gerçek veri ve "
                "ortam genellemesi değildir.",
                "Bir koşulda ön LiDAR ve GPS "
                "birlikte bozulur; sistem bunu "
                "tek kaynağa zorlamaz.",
            ],
        },
    )

    print("GPS hata izolasyonu: PASS")
    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()