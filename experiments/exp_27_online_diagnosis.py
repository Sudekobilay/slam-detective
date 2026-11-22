"""EXP-27: Ölçüm -> anomali -> hareket kararı -> yeni ölçüm çevrimi.

Tetikleme saati algoritmaya verilmez. Her komuttan sonra dünya bir adım ilerler;
gelecek ölçümler henüz üretilmemişken bir sonraki komut seçilir.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from estimation.online_diagnostic_agent import OnlineDiagnosticAgent
from storage.files import write_json, write_jsonl
from evaluation.trajectory import evaluate
from storage.files import write_csv
from simulator.streaming_world import StreamingWorld, WorldConfig


def drive_case(yaw_error_deg, strategy='adaptive', fault_start_s=2.):
    """Dünya ve denetleyici yalnız ölçüm/komut arayüzüyle haberleşir."""
    world = StreamingWorld(WorldConfig(front_yaw_error_deg=yaw_error_deg, fault_start_s=fault_start_s))
    agent = OnlineDiagnosticAgent(strategy=strategy)
    measurements = world.initial_measurements()
    agent.process(measurements)
    command_log = []
    for _ in range(40):
        command = agent.command()
        command_log.append({
            'interval_start_s': agent.estimator.time_s,
            'mode_before_step': agent.mode,
            'left_rate_rad_s': command[0], 'right_rate_rad_s': command[1],
            'differs_from_normal_route': command != agent.normal_command,
        })
        # Bu çağrıdan önce sonraki ölçüm paketi mevcut değildir.
        batch = world.step(*command)
        agent.process(batch)
        measurements.extend(batch)
    # Bu kayıtlar yalnız bütün kararlar tamamlandıktan sonra değerlendirilir.
    evaluation_records = world.evaluation_records()
    return agent, measurements, command_log, evaluation_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fault-yaw-deg', type=float, nargs='+', default=[2., 5.])
    parser.add_argument('--fault-start-s', type=float, default=2.)
    parser.add_argument('--strategy', choices=['adaptive', 'straight', 'passive'], default='adaptive')
    args = parser.parse_args()
    if any(not math.isfinite(value) or not 0 < abs(value) <= 10 for value in args.fault_yaw_deg):
        parser.error('Hata açılarının mutlak değeri 0 ile 10° arasında olmalı; sıfır hariç.')
    if not math.isfinite(args.fault_start_s) or not 0 <= args.fault_start_s <= 4:
        parser.error('Hata başlangıcı 0–4 saniye arasında olmalı.')
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'results' / f'exp_27_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print(f'EXP-27: NIS tetiklemeli adım adım simülasyon; strateji={args.strategy}; GPS güncellemesi kapalı.')
    cases = [('healthy', 0.)] + [
        (f'front_yaw_{yaw:g}deg'.replace('-', 'minus').replace('.', 'p'), yaw)
        for yaw in dict.fromkeys(args.fault_yaw_deg)
    ]
    summaries, evaluated_cases, truth_cases = [], {}, {}
    for name, yaw in cases:
        agent, events, commands, records = drive_case(yaw, args.strategy, args.fault_start_s)
        folder = output / name
        folder.mkdir()
        write_jsonl(folder / 'measurements.jsonl', events)
        truth_path = folder / 'ground_truth.csv'
        write_csv(truth_path, records['truth'])
        write_csv(folder / 'commands.csv', commands)
        write_csv(folder / 'agent_history.csv', agent.history)
        write_jsonl(folder / 'lidar_innovations.jsonl', agent.innovations)
        write_jsonl(folder / 'mounting_hypotheses.jsonl', agent.hypotheses)
        write_json(folder / 'decision.json', agent.decision)
        write_json(folder / 'simulation_metadata.json', records['metadata'])
        _, evaluated, metrics = evaluate({'line_slam': agent.path}, truth_path)
        write_csv(folder / 'line_slam.csv', evaluated['line_slam'])
        evaluated_cases[name], truth_cases[name] = evaluated['line_slam'], records['truth']
        summary = {
            'case': name,
            'anomaly_detected': agent.hold.active,
            'trigger_time_s': agent.hold.trigger_time_s,
            'selected_motion': agent.selected_motion,
            'final_mode': agent.mode,
            'full_position_rmse_m': metrics['line_slam']['position_rmse_m'],
            'full_heading_rmse_rad': metrics['line_slam']['heading_rmse_rad'],
            'intervals_changed_from_normal_route': sum(row['differs_from_normal_route'] for row in commands),
            'diagnostic_command_intervals': sum(row['mode_before_step'] == 'DIAGNOSING' for row in commands),
            'diagnosis_window_completed': agent.window_end_s is not None and agent.estimator.time_s >= agent.window_end_s - 1e-9,
            'held_lidar_scans': agent.counts['held_scans'],
            'line_updates': agent.estimator.line_updates,
            'landmark_count': agent.estimator.landmark_count,
            # Aşağıdaki iki alan yalnız çevrimdışı deney etiketinden türetilir.
            'alarm_on_healthy_case': yaw == 0. and agent.hold.active,
            'fault_present_without_alarm': yaw != 0. and not agent.hold.active,
        }
        summaries.append(summary)
        write_json(folder / 'summary.json', {**summary, 'trigger_evidence': agent.hold.trigger_evidence})
        print(
            f"{name}: alarm={agent.hold.active} | tetik={agent.hold.trigger_time_s} | "
            f"hareket={agent.selected_motion} | tüm rota konum RMSE={summary['full_position_rmse_m']:.6f} m | "
            f"normal rotadan farklı aralık={summary['intervals_changed_from_normal_route']}"
        )
    write_csv(output / 'comparison.csv', summaries)
    write_json(output / 'summary.json', {
        'experiment_id': 'exp_27', 'strategy': args.strategy,
        'nis_threshold': 9.21034, 'cases': summaries,
        'limitations': [
            'NIS alarmı hata kaynağının kesin teşhisi değildir.',
            'Bireysel eşik kullanılır; genel yanlış alarm ve kaçırma oranları kalibre edilmemiştir.',
            'Düşük genlikli veya eşleşmeyen hatalar tetiklenmeyebilir; bunlar raporlanır.',
            'Seçici yaklaşık geometrik ayrım skorunu kullanır; üstünlük garantisi yoktur.',
            'Seçilen hareket normal rota ile aynı olabilir; değiştirilmiş komut aralıkları ayrıca sayılır.',
            'LiDAR kilidi otomatik açılmaz; iki saniyelik teşhis penceresinden sonra komut sıfırlanır.',
            'Yetersiz harita desteği varsa hareket seçilmez ve sıfır komut kullanılır.',
            'GPS ölçümleri kaydedilir; bu deneyde GPS güncellemesi veya GPS hata hipotezi yoktur.',
            'Tek ortam ve tohum; üç koşul genel başarım oranı değildir.',
        ],
    })
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, rows in evaluated_cases.items():
        axes[0].plot([r['x_m'] for r in truth_cases[name]], [r['y_m'] for r in truth_cases[name]], label=name)
        axes[1].plot([r['time_s'] for r in rows], [r['position_error_m'] for r in rows], label=name)
    axes[0].set(title='Uygulanan komutların gerçek rotası', xlabel='x (m)', ylabel='y (m)')
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[1].set(title='Çevrimdışı konum hatası', xlabel='Zaman (s)', ylabel='Hata (m)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'comparison.png', dpi=160)
    plt.close(figure)
    print(f'Sonuç klasörü: {output}')


if __name__ == '__main__':
    main()
