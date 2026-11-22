"""EXP-40: JSON sensör düzeniyle çevrim içi teşhis ve hareket seçimi."""

from application.layout_service import create_agent, calibrate_layout, run_layout

from simulator.layout_world import LayoutWorld
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
import json
import math
from pathlib import Path
import sys

from estimation.active_pipeline import ActivePipeline, make_backend, geometry_view
from estimation.configurable_geometry import (
    LayoutScanGuard, compare_layout_hypotheses, select_layout_motion, usable_alignments,
)
from estimation.diagnostic_motion import CANDIDATES
from estimation.scan_alignment_guard import ScanGuardConfig, scan_alignment
from storage.files import write_json, write_jsonl
from configuration.layouts import read_layout
from evaluation.trajectory import evaluate
from storage.files import write_csv
from simulator.encoder import EncoderMeasurement
from simulator.imu import GyroMeasurement
from simulator.measurement_stream import measurement_to_event
from simulator.multi_lidar import LidarUnit
from simulator.streaming_world import StreamingWorld, WorldConfig










def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Yapılandırma verilmezse doğrulanmış EXP-39 davranışı korunur.
    if not any(arg == '--config' or arg.startswith('--config=') for arg in argv) and not any(arg in ('--help', '-h') for arg in argv):
        from experiments.exp_39_active_pipeline import main as legacy_main
        return legacy_main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--algorithm', choices=['ekf', 'fastslam1', 'fastslam2'], default='ekf')
    parser.add_argument('--policy', choices=['adaptive', 'passive', 'fixed_straight', 'fixed_arc'], default='adaptive')
    parser.add_argument('--normal', choices=['straight', 'arc'], default='straight')
    parser.add_argument('--fault', choices=['healthy', 'lidar', 'gps', 'combined'], default='lidar')
    parser.add_argument('--fault-sensor')
    parser.add_argument('--yaw', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--particles', type=int, default=40)
    gps = parser.add_mutually_exclusive_group()
    gps.add_argument('--gps', dest='gps_override', action='store_const', const=True)
    gps.add_argument('--no-gps', dest='gps_override', action='store_const', const=False)
    args = parser.parse_args(argv)
    data = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.gps_override is not None:
        data['gps_enabled'] = args.gps_override
    read_layout(data)
    if args.particles < 2:
        parser.error('En az iki parçacık gerekli.')
    # Hata ayarını pahalı kalibrasyondan önce doğrula.
    LayoutWorld(data, args.seed, args.fault,
                args.fault_sensor or data['lidars'][0]['id'], args.yaw)
    config, calibration = calibrate_layout(data, args.algorithm, args.particles)
    agent, events, commands, records, reason = run_layout(
        data, config, algorithm=args.algorithm, policy=args.policy, normal=args.normal,
        fault=args.fault, sensor=args.fault_sensor, yaw=args.yaw,
        seed=args.seed, particles=args.particles,
    )
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'results' / f'exp_40_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'used_config.json', data)
    write_json(output / 'calibration.json', calibration)
    write_json(output / 'simulation_metadata.json', records['metadata'])
    write_json(output / 'decision.json', agent.decision)
    write_jsonl(output / 'measurements.jsonl', events)
    write_jsonl(output / 'raw_hypotheses.jsonl', agent.raw_history)
    write_jsonl(output / 'scan_guard.jsonl', agent.guard.history)
    write_csv(output / 'commands.csv', commands)
    write_csv(output / 'agent_history.csv', agent.history)
    write_csv(output / 'gps_diagnostics.csv', agent.gps_monitor.history)
    truth = output / 'ground_truth.csv'
    write_csv(truth, records['truth'])
    _, evaluated, _ = evaluate({'reference': agent.reference_path, 'output': agent.path}, truth)
    for name, rows in evaluated.items():
        write_csv(output / f'{name}_evaluation.csv', rows)
    completed = reason is None and agent.time_s >= max(4., agent.window_end_s or 4.) - 1e-9
    summary = {
        'experiment_id': 'exp_40', 'algorithm': args.algorithm, 'layout': data['name'],
        'lidar_count': len(data['lidars']), 'trigger_time_s': agent.trigger_time_s,
        'selected_motion': agent.selected_motion, 'required_horizon_completed': completed,
        'stop_reason': reason, 'last_status': agent.history[-1]['joint_status'],
        'limitations': [
            'Tek LiDAR için geometrik kaynak ayrımı ve bu alarm üzerinden hareket seçimi desteklenmez.',
            'Her LiDAR için ayrı tek-montaj-hatası ve ortak yön hatası modelleri; eşzamanlı çoklu LiDAR hatası modellenmez.',
            'Ham model ve aday hareket için en az iki destekli sensör gerekir; destek yoksa seçim yapılmaz.',
            'Adaylar kısa düz ve yay hareketidir; bilinmeyen alan güvenliği veya genel rota planlama değildir.',
            'GPS-only adaptif hareket modeli henüz yoktur; GPS alarmında desteklenen geometrik hareket yoksa durulur.',
            'Ayrı sağlıklı kalibrasyon, tek ortam, ideal LiDAR. Çoklu çift alarmı olasılıksal güven değildir.',
        ],
    }
    write_json(output / 'summary.json', summary)
    print(f"Düzen: {data['name']} | LiDAR: {len(data['lidars'])} | algoritma: {args.algorithm}")
    print(f'Alarm: {agent.trigger_time_s} | seçilen hareket: {agent.selected_motion}')
    print(f'Tamamlandı: {completed} | son durum: {summary["last_status"]}')
    if len(data['lidars']) == 1:
        print('Tek LiDAR: sensörler arası kaynak ayrımı için destek yetersiz; bu bir sağlıklı kararı değildir.')
    print(f'Sonuç klasörü: {output}')
    if not completed:
        raise RuntimeError('Gerekli süre tamamlanmadı.')


if __name__ == '__main__':
    main()
