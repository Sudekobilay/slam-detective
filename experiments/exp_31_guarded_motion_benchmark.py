"""EXP-31: Sağlıklı kalibrasyonlu alarm ile çevrim içi hareket karşılaştırması.

EXP-29'un eşit süre protokolü kullanılır. Gerçek konum ve hata etiketi yalnız
simülasyon/değerlendirme katmanındadır. Hipotez farkı teşhis doğruluğu değildir.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
import math
from pathlib import Path

from estimation.diagnostic_motion import CANDIDATES
from estimation.online_diagnostic_agent import OnlineDiagnosticAgent
from estimation.scan_alignment_guard import ScanAlignmentGuard
from experiments.exp_24_motion_comparison import write_json, write_jsonl
from experiments.exp_28_policy_benchmark import POLICIES
from experiments.exp_29_complete_window import (
    BASE_HORIZON,
    drive_trial,
    position_rmse,
    window_evidence,
)
from experiments.exp_30_scan_guard_audit import (
    CALIBRATION_SEEDS,
    calibrate_healthy,
)
from experiments.line_slam_common import evaluate, write_csv


class GuardedFixedArcAgent(OnlineDiagnosticAgent):
    def __init__(self, config):
        super().__init__(
            strategy="passive",
            scan_guard=ScanAlignmentGuard(config),
        )

    def process(self, events):
        was_active = self.hold.active
        super().process(events)

        if not was_active and self.hold.active:
            self.selected_motion = "arc"
            self.mode = "DIAGNOSING"
            self.history[-1]["mode"] = self.mode

    def command(self):
        if (
            self.hold.active
            and self.estimator.time_s < self.window_end_s - 1e-9
        ):
            return CANDIDATES["arc"]

        return super().command()


def make_guarded_agent(policy, normal_motion, *, config):
    if policy not in POLICIES or normal_motion not in CANDIDATES:
        raise ValueError("Geçersiz politika veya normal rota.")

    if policy == "fixed_arc":
        agent = GuardedFixedArcAgent(config)
    else:
        strategy = {
            "passive": "passive",
            "fixed_straight": "straight",
            "adaptive": "adaptive",
        }[policy]

        agent = OnlineDiagnosticAgent(
            strategy=strategy,
            scan_guard=ScanAlignmentGuard(config),
        )

    agent.normal_command = CANDIDATES[normal_motion]
    return agent


def check_pairing(trials):
    """Ölçüm geçmişi, ilk alarm ve pencere sonunu eşleştirir."""
    first = next(iter(trials.values()))
    trigger = first[0].hold.trigger_time_s
    cutoff = BASE_HORIZON if trigger is None else trigger

    prefix = [
        event
        for event in first[1]
        if event["time_s"] <= cutoff + 1e-9
    ]

    return {
        "same_first_alarm": all(
            trial[0].hold.trigger_time_s == trigger
            for trial in trials.values()
        ),
        "same_window_end": all(
            trial[0].window_end_s == first[0].window_end_s
            for trial in trials.values()
        ),
        "same_measurements_through_alarm": all(
            [
                event
                for event in trial[1]
                if event["time_s"] <= cutoff + 1e-9
            ] == prefix
            for trial in trials.values()
        ),
    }


def movement_metrics(truth, commands, trigger, end):
    """Çevrimdışı hareket maliyeti; alarm anı başlangıca dahildir."""
    if trigger is None:
        return {
            key: None
            for key in (
                "diagnostic_distance_m",
                "diagnostic_absolute_turn_rad",
                "changed_command_duration_s",
                "diagnostic_duration_s",
            )
        }

    points = [
        point
        for point in truth
        if trigger - 1e-9 <= point["time_s"] <= end + 1e-9
    ]

    distance = sum(
        math.hypot(
            second["x_m"] - first["x_m"],
            second["y_m"] - first["y_m"],
        )
        for first, second in zip(points, points[1:])
    )

    turn = sum(
        abs(
            math.atan2(
                math.sin(second["theta_rad"] - first["theta_rad"]),
                math.cos(second["theta_rad"] - first["theta_rad"]),
            )
        )
        for first, second in zip(points, points[1:])
    )

    duration = sum(
        min(
            commands[index + 1]["interval_start_s"]
            if index + 1 < len(commands)
            else truth[-1]["time_s"],
            end,
        ) - command["interval_start_s"]
        for index, command in enumerate(commands)
        if (
            command["applied"]
            and command["changed_from_normal"]
            and trigger - 1e-9
            <= command["interval_start_s"]
            < end - 1e-9
        )
    )

    return {
        "diagnostic_distance_m": distance,
        "diagnostic_absolute_turn_rad": turn,
        "changed_command_duration_s": duration,
        "diagnostic_duration_s": (
            max(0.0, points[-1]["time_s"] - trigger)
            if points
            else 0.0
        ),
    }


def mean_defined(rows, key):
    values = [
        row[key]
        for row in rows
        if row[key] is not None
    ]
    return sum(values) / len(values) if values else None


def summarize(rows):
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["normal_motion"],
            row["policy"],
            row["fault_yaw_deg"],
        )
        groups[key].append(row)

    output = []

    for (normal, policy, yaw), trials in sorted(groups.items()):
        windows = [
            row
            for row in trials
            if row["diagnostic_window_completed"]
        ]

        support = sum(
            row["supported_frames"]
            for row in windows
        )

        pairs = [
            row
            for row in windows
            if row["margin_difference_from_passive"] is not None
        ]

        output.append({
            "normal_motion": normal,
            "policy": policy,
            "fault_yaw_deg": yaw,
            "trials": len(trials),
            "alarm_trials": sum(row["alarm"] for row in trials),
            "healthy_alarm_trials": sum(
                row["alarm"] and yaw == 0
                for row in trials
            ),
            "premature_alarm_trials": sum(
                row["premature_alarm"]
                for row in trials
            ),
            "complete_windows": len(windows),
            "completed_trials": sum(
                row["required_horizon_completed"]
                for row in trials
            ),
            "mean_detection_delay_s": mean_defined(
                trials, "detection_delay_s"
            ),
            "mean_common_position_rmse_m": mean_defined(
                trials, "common_position_rmse_m"
            ),
            "mean_diagnostic_position_rmse_m": mean_defined(
                windows, "diagnostic_position_rmse_m"
            ),
            "supported_frames": support,
            "window_scan_instants": sum(
                row["window_scan_instants"]
                for row in windows
            ),
            "pooled_front_margin": (
                sum(row["front_margin_sum"] for row in windows)
                / support
                if support
                else None
            ),
            "paired_margin_trials": len(pairs),
            "mean_margin_difference_from_passive": mean_defined(
                pairs, "margin_difference_from_passive"
            ),
            "trials_with_changed_commands": sum(
                (row["changed_command_duration_s"] or 0) > 1e-9
                for row in trials
            ),
            "selected_straight_trials": sum(
                row["selected_motion"] == "straight"
                for row in trials
            ),
            "selected_arc_trials": sum(
                row["selected_motion"] == "arc"
                for row in trials
            ),
            "mean_diagnostic_distance_m": mean_defined(
                windows, "diagnostic_distance_m"
            ),
            "mean_absolute_turn_rad": mean_defined(
                windows, "diagnostic_absolute_turn_rad"
            ),
            "mean_changed_command_duration_s": mean_defined(
                windows, "changed_command_duration_s"
            ),
            "mean_extra_distance_from_passive_m": mean_defined(
                windows, "extra_distance_from_passive_m"
            ),
            "mean_endpoint_deviation_from_passive_m": mean_defined(
                windows, "endpoint_deviation_from_passive_m"
            ),
            "mean_extra_simulation_time_s": mean_defined(
                trials, "extra_simulation_time_s"
            ),
        })

    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[401, 402, 403, 404, 405],
    )
    args = parser.parse_args()

    if (
        len(set(args.seeds)) != len(args.seeds)
        or set(args.seeds) & set(CALIBRATION_SEEDS)
    ):
        parser.error(
            "Tohumlar tekil ve kalibrasyon tohumlarından farklı olmalı."
        )

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_31_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    print(
        "EXP-31: sağlıklı kalibrasyon + çevrim içi dört politika; "
        "GPS güncellemesi kapalı.",
        flush=True,
    )

    config, calibration, samples = calibrate_healthy()
    write_json(output / "calibration.json", calibration)
    write_csv(output / "calibration_samples.csv", samples)

    factory = partial(make_guarded_agent, config=config)
    rows, pairing = [], {}

    for normal in ("straight", "arc"):
        for seed in args.seeds:
            for yaw in (0.0, 2.0, 5.0):
                condition = (
                    "healthy"
                    if yaw == 0
                    else f"front_yaw_{yaw:g}deg"
                )

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
                pairing[f"{normal}/{seed}/{condition}"] = checks

                if not all(checks.values()):
                    raise RuntimeError(
                        "Politikaların alarm öncesi geçmişi "
                        "veya alarm zamanı farklı."
                    )

                case_rows = []

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
                        / condition
                        / policy
                    )
                    folder.mkdir(parents=True)

                    write_jsonl(
                        folder / "measurements.jsonl", events
                    )

                    truth_path = folder / "ground_truth.csv"
                    write_csv(truth_path, records["truth"])
                    write_csv(folder / "commands.csv", commands)
                    write_csv(
                        folder / "agent_history.csv",
                        agent.history,
                    )
                    write_jsonl(
                        folder / "scan_alignment.jsonl",
                        agent.scan_guard.history,
                    )
                    write_jsonl(
                        folder / "lidar_innovations.jsonl",
                        agent.innovations,
                    )
                    write_jsonl(
                        folder / "mounting_hypotheses.jsonl",
                        agent.hypotheses,
                    )
                    write_json(
                        folder / "decision.json", agent.decision
                    )
                    write_json(
                        folder / "simulation_metadata.json",
                        records["metadata"],
                    )

                    _, evaluated, _ = evaluate(
                        {"line_slam": agent.path},
                        truth_path,
                    )
                    estimated = evaluated["line_slam"]
                    write_csv(
                        folder / "line_slam.csv", estimated
                    )

                    trigger = agent.hold.trigger_time_s
                    end = agent.window_end_s
                    evidence = window_evidence(
                        agent.hypotheses, trigger, end
                    )
                    complete = horizon[
                        "diagnostic_window_completed"
                    ]

                    if (
                        complete
                        and evidence["window_scan_instants"] != 4
                    ):
                        raise RuntimeError(
                            "Tam pencerede dört tarama anı bekleniyor."
                        )

                    diagnostic = [
                        row
                        for row in estimated
                        if (
                            trigger is not None
                            and trigger + 1e-9
                            < row["time_s"]
                            <= end + 1e-9
                        )
                    ]

                    premature = (
                        yaw != 0
                        and trigger is not None
                        and trigger < 2.0 - 1e-9
                    )

                    common = [
                        row
                        for row in estimated
                        if row["time_s"] <= BASE_HORIZON + 1e-9
                    ]

                    row = {
                        "normal_motion": normal,
                        "seed": seed,
                        "fault_yaw_deg": yaw,
                        "policy": policy,
                        "alarm": agent.hold.active,
                        "trigger_time_s": trigger,
                        "trigger_source": agent.hold.trigger_source,
                        "premature_alarm": premature,
                        "detection_delay_s": (
                            trigger - 2.0
                            if (
                                yaw != 0
                                and trigger is not None
                                and not premature
                            )
                            else None
                        ),
                        "selected_motion": agent.selected_motion,
                        **horizon,
                        "stop_reason": reason,
                        "simulation_end_s": agent.estimator.time_s,
                        "extra_simulation_time_s": max(
                            0.0,
                            agent.estimator.time_s - BASE_HORIZON,
                        ),
                        "common_position_rmse_m": (
                            position_rmse(common)
                            if horizon["common_horizon_completed"]
                            else None
                        ),
                        "diagnostic_position_rmse_m": (
                            position_rmse(diagnostic)
                            if complete
                            else None
                        ),
                        "quarantined_scans": (
                            agent.counts["quarantined_scans"]
                        ),
                        **evidence,
                        **movement_metrics(
                            records["truth"],
                            commands,
                            trigger,
                            end,
                        ),
                        "extra_distance_from_passive_m": None,
                        "endpoint_deviation_from_passive_m": None,
                        "margin_difference_from_passive": None,
                    }

                    case_rows.append(row)

                passive = case_rows[0]
                passive_truth = trials["passive"][3]["truth"]

                for row in case_rows:
                    if (
                        row["diagnostic_window_completed"]
                        and passive["diagnostic_window_completed"]
                    ):
                        endpoint = trials[
                            row["policy"]
                        ][3]["truth"][-1]
                        reference = passive_truth[-1]

                        row[
                            "endpoint_deviation_from_passive_m"
                        ] = math.hypot(
                            endpoint["x_m"] - reference["x_m"],
                            endpoint["y_m"] - reference["y_m"],
                        )

                        row["extra_distance_from_passive_m"] = (
                            row["diagnostic_distance_m"]
                            - passive["diagnostic_distance_m"]
                        )

                        # Kapsamlar farklıysa ortalamalar
                        # doğrudan karşılaştırılmaz.
                        if (
                            row["supported_frames"]
                            == passive["supported_frames"]
                            == 4
                        ):
                            row[
                                "margin_difference_from_passive"
                            ] = (
                                row["mean_front_margin"]
                                - passive["mean_front_margin"]
                            )

                    folder = (
                        output
                        / f"normal_{normal}"
                        / f"seed_{seed}"
                        / condition
                        / row["policy"]
                    )

                    write_json(
                        folder / "summary.json",
                        {
                            **row,
                            "trigger_evidence": trials[
                                row["policy"]
                            ][0].hold.trigger_evidence,
                        },
                    )

                rows.extend(case_rows)

            print(
                f"{normal}/seed_{seed}: tamamlandı "
                f"({len(rows)} deneme).",
                flush=True,
            )

    groups = summarize(rows)
    write_csv(output / "trials.csv", rows)
    write_csv(output / "group_summary.csv", groups)

    write_json(
        output / "summary.json",
        {
            "experiment_id": "exp_31",
            "trial_count": len(rows),
            "seeds": args.seeds,
            "calibration": calibration,
            "pairing_checks": pairing,
            "groups": groups,
            "all_required_horizons_completed": all(
                row["required_horizon_completed"]
                for row in rows
            ),
            "protocol": {
                "normal_horizon_s": 4.0,
                "diagnostic_duration_s": 2.0,
                "diagnostic_interval": "(alarm,alarm+2]",
            },
            "limitations": [
                "Alarm, hata kaynağı teşhisi değildir; "
                "hipotez farkı kalibre edilmiş doğruluk/olasılık değildir.",
                "İki saniyelik pencerede eksik hipotez desteği "
                "raporlanır; sıfır kanıt olarak doldurulmaz.",
                "Eşleştirilmiş hipotez farkı yalnız iki yöntemde "
                "de 4/4 destek varsa hesaplanır.",
                "Seed 401–405 önceki alarm doğrulamasında kullanıldı; "
                "bu çalışma yeni, bağımsız ortam testi değildir.",
                "İdeal LiDAR, tek ortam, yalnız ön LiDAR montaj "
                "hatası; GPS yalnız kaydedilir.",
                "Hareket seçici yerel geometrik yaklaşık puan "
                "kullanır; üstünlüğü varsayılmaz.",
                "Mesafe nokta arası kirişlerin toplamıdır; "
                "farklı rota için küçük sayısal farklar olabilir.",
                "Ek simülasyon süresi protokol maliyetidir; "
                "görev tamamlanma gecikmesi ölçülmez.",
                "Hesaplama gecikmesi ve gerçek zamanlılık "
                "bu deneyde ölçülmez.",
            ],
        },
    )

    print("Ortak geçmiş ve ilk alarm eşleştirmeleri: PASS")
    print(
        "Gerekli süreler tamamlandı: "
        f"{all(row['required_horizon_completed'] for row in rows)}"
    )

    for group in groups:
        print(
            f"{group['normal_motion']}/"
            f"{group['fault_yaw_deg']:g}deg/"
            f"{group['policy']}: "
            f"alarm={group['alarm_trials']}/{group['trials']} | "
            f"destek={group['supported_frames']}/"
            f"{group['window_scan_instants']} | "
            f"hipotez farkı={group['pooled_front_margin']} | "
            f"hareket değişen="
            f"{group['trials_with_changed_commands']}"
        )

    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()