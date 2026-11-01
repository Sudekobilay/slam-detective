"""EXP-40: JSON sensör düzeniyle çevrim içi teşhis ve hareket seçimi."""
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
from experiments.exp_24_motion_comparison import write_json, write_jsonl
from experiments.exp_36_sensor_layouts import read_layout
from experiments.line_slam_common import evaluate, write_csv
from simulator.encoder import EncoderMeasurement
from simulator.imu import GyroMeasurement
from simulator.measurement_stream import measurement_to_event
from simulator.multi_lidar import LidarUnit
from simulator.streaming_world import StreamingWorld, WorldConfig


class LayoutWorld(StreamingWorld):
    """Gerçek konum ve gerçek montaj yalnız simülasyon üretiminde kullanılır."""
    def __init__(self, data, seed, fault='healthy', sensor=None, yaw=5.):
        self.units = read_layout(data)
        ids = {unit.sensor_id for unit in self.units}
        if fault not in ('healthy', 'lidar', 'gps', 'combined'):
            raise ValueError('Geçersiz hata türü.')
        if fault in ('lidar', 'combined') and sensor not in ids:
            raise ValueError('Hatalı sensör yapılandırmada bulunamadı.')
        if fault in ('gps', 'combined') and not data['gps_enabled']:
            raise ValueError('GPS kapalıyken GPS hatası uygulanamaz.')
        if not math.isfinite(yaw) or abs(yaw) > 10:
            raise ValueError('Montaj hatası en çok 10 derece olmalı.')
        self.data, self.fault, self.fault_sensor, self.yaw = data, fault, sensor, yaw
        super().__init__(WorldConfig(gyro_seed=seed, gps_seed=seed + 1))

    def _batch(self, motion_packets):
        base = super()._batch(motion_packets)
        events = [e for e in base if e['sensor_type'] != 'lidar'
                  and (self.data['gps_enabled'] or e['sensor_type'] != 'gps')]
        time = round(self._index * self.dt, 12)
        if self._index % 5 == 0:
            for nominal in self.units:
                unit = nominal
                if time >= 2. and self.fault in ('lidar', 'combined') and nominal.sensor_id == self.fault_sensor:
                    mount = replace(nominal.actual_mount,
                                    yaw=nominal.actual_mount.yaw + math.radians(self.yaw))
                    unit = LidarUnit(nominal.sensor_id, mount, nominal.config)
                events.append(measurement_to_event(unit.measure(self._robot.pose, self._environment, time)))
        if time >= 2. and self.fault in ('gps', 'combined'):
            for event in events:
                if event['sensor_type'] == 'gps' and event['data']['available']:
                    event['data']['x_m'] += .5
                    event['data']['y_m'] += .3
        events.sort(key=lambda e: (e['time_s'], e['sensor_type'], e['sensor_id']))
        return json.loads(json.dumps(events, allow_nan=False))

    def evaluation_records(self):
        records = super().evaluation_records()
        records['metadata']['layout'] = self.data
        records['metadata']['layout_fault'] = {
            'type': self.fault, 'sensor': self.fault_sensor,
            'yaw_deg': self.yaw if self.fault in ('lidar', 'combined') else 0.,
            'start_s': 2., 'gps_bias_m': [.5, .3] if self.fault in ('gps', 'combined') else [0., 0.],
        }
        return records


def create_agent(data, config, algorithm='ekf', **options):
    units = read_layout(data)
    mounts = {u.sensor_id: u.actual_mount for u in units}
    configs = {u.sensor_id: u.config for u in units}
    return ActivePipeline(
        algorithm, config, mounts=mounts, guard=LayoutScanGuard(config),
        raw_compare=compare_layout_hypotheses,
        motion_selector=partial(select_layout_motion, lidar_configs=configs), **options,
    )


def calibrate_layout(data, algorithm='ekf', particles=40):
    units = read_layout(data)
    mounts = {u.sensor_id: u.actual_mount for u in units}
    collection = ScanGuardConfig(19., 100.)
    gaps, rms = [], []
    for normal in CANDIDATES:
        for seed in range(11, 16):
            world = LayoutWorld(data, seed)
            backend = make_backend(algorithm, particles, 10000 + seed, mounts)
            batch = world.initial_measurements()
            for index in range(41):
                if index:
                    batch = world.step(*CANDIDATES[normal])
                    encoder = next(e for e in batch if e['sensor_type'] == 'encoder')
                    gyro = next(e for e in batch if e['sensor_type'] == 'gyro')
                    backend.predict(EncoderMeasurement(time_s=encoder['time_s'], **encoder['data']),
                                    GyroMeasurement(time_s=gyro['time_s'], **gyro['data']))
                packets = [e for e in batch if e['sensor_type'] == 'lidar']
                if packets:
                    view = geometry_view(backend)
                    alignments = {e['sensor_id']: scan_alignment(
                        view, e, mounts[e['sensor_id']], collection
                    ) for e in packets}
                    good = usable_alignments(alignments, collection)
                    rms.extend(row['alignment_rms_m'] for row in good.values())
                    if len(good) >= 2:
                        offsets = [row['offset_deg'] for row in good.values()]
                        gaps.append(max(offsets) - min(offsets))
                    for event in packets:
                        backend.update_scan(event['sensor_id'], event['data']['scan']['points'])
            print(f'Kalibrasyon: {data["name"]}/{algorithm}/{normal}/seed_{seed}', flush=True)
    if not rms or (len(units) >= 2 and not gaps):
        raise RuntimeError('Bu yerleşimde sağlıklı kalibrasyon desteği yetersiz.')
    config = ScanGuardConfig(max(gaps, default=0.) + .5, max(rms) + .01)
    return config, {'layout': data, 'algorithm': algorithm, 'particles': particles,
                    'healthy_sequences': 10, 'sensor_seeds': list(range(11, 16)),
                    'supported_pair_frames': len(gaps), 'config': asdict(config)}


def run_layout(data, config, *, algorithm='ekf', policy='adaptive', normal='straight',
               fault='lidar', sensor=None, yaw=5., seed=1001, particles=40):
    units = read_layout(data)
    sensor = units[0].sensor_id if sensor is None else sensor
    world = LayoutWorld(data, seed, fault, sensor, yaw)
    agent = create_agent(data, config, algorithm, policy=policy, normal=normal,
                         seed=seed + 20000, particles=particles, use_gps=data['gps_enabled'])
    events = world.initial_measurements()
    agent.process(events)
    commands, reason = [], None
    deadline = 4.
    while agent.time_s < deadline - 1e-9:
        left, right = agent.command()
        command = {'interval_start_s': agent.time_s, 'left_rate_rad_s': left,
                   'right_rate_rad_s': right, 'mode': agent.mode, 'applied': False}
        commands.append(command)
        try:
            batch = world.step(left, right)
        except RuntimeError as error:
            reason = str(error)
            break
        command['applied'] = True
        agent.process(batch)
        events.extend(batch)
        deadline = max(4., agent.window_end_s or 4.)
    return agent, events, commands, world.evaluation_records(), reason


def main():
    # Yapılandırma verilmezse doğrulanmış EXP-39 davranışı korunur.
    if not any(arg == '--config' or arg.startswith('--config=') for arg in sys.argv[1:]) and not any(arg in ('--help', '-h') for arg in sys.argv[1:]):
        from experiments.exp_39_active_pipeline import main as legacy_main
        return legacy_main()
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
    args = parser.parse_args()
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
