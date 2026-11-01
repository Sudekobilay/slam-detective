"""EXP-33: Hipotezlerin tahmini taramalarına göre hareket seçimi."""

from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from estimation.hypothesis_motion import (
    HypothesisMotionAgent,
)
from experiments.exp_24_motion_comparison import (
    write_json,
    write_jsonl,
)
from experiments.exp_28_policy_benchmark import POLICIES
from experiments.exp_29_complete_window import (
    drive_trial,
    position_rmse,
)
from experiments.exp_30_scan_guard_audit import (
    calibrate_healthy,
)
from experiments.exp_31_guarded_motion_benchmark import (
    check_pairing,
    movement_metrics,
)
from experiments.line_slam_common import evaluate, write_csv


def make_agent(policy, normal, *, config):
    return HypothesisMotionAgent(
        policy,
        normal,
        config,
    )


def raw_window(history, trigger, end):
    frames = [
        row
        for row in history
        if (
            trigger is not None
            and trigger + 1e-9
            < row["time_s"]
            <= end + 1e-9
        )
    ]

    supported = [
        row
        for row in frames
        if row["status"] == "evaluated"
    ]

    margins = []

    for row in supported:
        scores = {
            model["hypothesis"]: model["cost_m2"]
            for model in row["models"]
        }

        margins.append(
            min(
                scores["rear_yaw"],
                scores["heading_offset"],
            ) - scores["front_yaw"]
        )

    return {
        "window_frames": len(frames),
        "supported_frames": len(supported),
        "mean_front_margin_m2": (
            sum(margins) / len(margins)
            if margins
            else None
        ),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = root / "results" / f"exp_33_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    print(
        "EXP-33: hipoteze bağlı çevrim içi hareket seçimi; "
        "GPS güncellemesi kapalı.",
        flush=True,
    )

    config, calibration, _ = calibrate_healthy()
    write_json(
        output / "calibration.json",
        calibration,
    )

    factory = partial(make_agent, config=config)
    rows = []
    pairing = {}

    for normal in ("straight", "arc"):
        for seed in (501, 502, 503, 504, 505):
            for yaw in (0.0, 2.0, 5.0):
                trials = {
                    policy: drive_trial(
                        policy,
                        normal,
                        yaw,
                        seed,
                        agent_factory=factory,
                    )
                    for policy in POLICIES
                }

                checks = check_pairing(trials)
                pairing[
                    f"{normal}/{seed}/{yaw:g}"
                ] = checks

                if not all(checks.values()):
                    raise RuntimeError(
                        "Alarm veya ortak geçmiş eşleşmiyor."
                    )

                for policy, trial in trials.items():
                    (
                        agent,
                        events,
                        commands,
                        records,
                        horizon,
                        reason,
                    ) = trial

                    folder = (
                        output
                        / f"normal_{normal}"
                        / f"seed_{seed}"
                        / f"yaw_{yaw:g}"
                        / policy
                    )
                    folder.mkdir(parents=True)

                    truth_path = (
                        folder / "ground_truth.csv"
                    )

                    write_csv(
                        truth_path,
                        records["truth"],
                    )
                    write_jsonl(
                        folder / "measurements.jsonl",
                        events,
                    )
                    write_csv(
                        folder / "commands.csv",
                        commands,
                    )
                    write_json(
                        folder / "decision.json",
                        agent.decision,
                    )
                    write_jsonl(
                        folder / "raw_hypotheses.jsonl",
                        agent.raw_history,
                    )
                    write_jsonl(
                        folder / "scan_guard.jsonl",
                        agent.scan_guard.history,
                    )

                    _, evaluated, _ = evaluate(
                        {"line_slam": agent.path},
                        truth_path,
                    )

                    path = evaluated["line_slam"]

                    write_csv(
                        folder / "line_slam.csv",
                        path,
                    )

                    common = [
                        row
                        for row in path
                        if row["time_s"] <= 4.0 + 1e-9
                    ]

                    row = {
                        "normal_motion": normal,
                        "seed": seed,
                        "fault_yaw_deg": yaw,
                        "policy": policy,
                        "alarm": agent.hold.active,
                        "trigger_time_s": (
                            agent.hold.trigger_time_s
                        ),
                        "selected_motion": (
                            agent.selected_motion
                        ),
                        **horizon,
                        "stop_reason": reason,
                        "common_position_rmse_m": (
                            position_rmse(common)
                            if horizon[
                                "common_horizon_completed"
                            ]
                            else None
                        ),
                        **raw_window(
                            agent.raw_history,
                            agent.hold.trigger_time_s,
                            agent.window_end_s,
                        ),
                        **movement_metrics(
                            records["truth"],
                            commands,
                            agent.hold.trigger_time_s,
                            agent.window_end_s,
                        ),
                    }

                    rows.append(row)
                    write_json(
                        folder / "summary.json",
                        row,
                    )

            print(
                f"{normal}/seed_{seed}: "
                f"{len(rows)} deneme tamamlandı.",
                flush=True,
            )

    groups = defaultdict(list)

    for row in rows:
        groups[
            row["normal_motion"],
            row["fault_yaw_deg"],
            row["policy"],
        ].append(row)

    summary = []

    for (normal, yaw, policy), trials in sorted(
        groups.items()
    ):
        supported = [
            row
            for row in trials
            if (
                row["diagnostic_window_completed"]
                and row["supported_frames"] == 4
            )
        ]

        entry = {
            "normal_motion": normal,
            "fault_yaw_deg": yaw,
            "policy": policy,
            "trials": len(trials),
            "alarm_trials": sum(
                row["alarm"] for row in trials
            ),
            "complete_supported_trials": (
                len(supported)
            ),
            "mean_front_margin_m2": (
                sum(
                    row["mean_front_margin_m2"]
                    for row in supported
                ) / len(supported)
                if supported
                else None
            ),
            "selected_straight": sum(
                row["selected_motion"] == "straight"
                for row in trials
            ),
            "selected_arc": sum(
                row["selected_motion"] == "arc"
                for row in trials
            ),
            "changed_command_trials": sum(
                (
                    row["changed_command_duration_s"]
                    or 0
                ) > 1e-9
                for row in trials
            ),
        }

        summary.append(entry)

        print(
            f"{normal}/{yaw:g}deg/{policy}: "
            f"alarm={entry['alarm_trials']}/5 | "
            f"tam destek={len(supported)}/5 | "
            f"ön fark={entry['mean_front_margin_m2']} | "
            f"düz/yay={entry['selected_straight']}/"
            f"{entry['selected_arc']}"
        )

    write_csv(output / "trials.csv", rows)
    write_csv(output / "group_summary.csv", summary)

    write_json(
        output / "summary.json",
        {
            "experiment_id": "exp_33",
            "groups": summary,
            "pairing_checks": pairing,
            "all_horizons_completed": all(
                row["required_horizon_completed"]
                for row in rows
            ),
            "limitations": [
                "Puanlar geometrik MSE farkıdır; "
                "başarı yüzdesi veya olasılık değildir.",
                "Seçici o andaki en iyi iki modelin "
                "sabit açı tahminlerini kullanır; "
                "belirsizlik dağılımı yayılmaz.",
                "Tahmini menzil farkları .5 m ile "
                "kırpılır; ortak menzil ışınları kullanılır.",
                "Her aday dört gelecekteki tarama anında "
                "iki sensörde de en az 20 ortak ışın "
                "gerektirir.",
                "Bilinen haritada açıklık kontrolü, "
                "bilinmeyen alan için güvenlik "
                "garantisi değildir.",
                "Yeni değerlendirme seedleri 501–505; "
                "aynı ortam ve ideal LiDAR. "
                "Ortam genellemesi değildir.",
                "Sadece ön LiDAR hatası değerlendirildi; "
                "GPS, arka hata ve ortak yön hatası "
                "saha değerlendirmesi yok.",
                "İlk alarm sonrası iki saniye; "
                "konum hatası ortak [0,4] ufkunda "
                "hesaplanır.",
                "Aktif yöntemin üstünlüğü varsayılmaz; "
                "her yöntem ayrı ham kayıt üretir.",
            ],
        },
    )

    print(
        "Ortak geçmiş ve alarm eşleştirmeleri: PASS"
    )
    print(
        "Gerekli süreler tamamlandı: "
        f"{all(row['required_horizon_completed'] for row in rows)}"
    )
    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()