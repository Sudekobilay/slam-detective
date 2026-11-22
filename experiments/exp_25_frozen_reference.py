"""EXP-25: Ortak başlangıçtan sonra sabit haritayla teşhis kontrolü.

İlk 2 saniyeden önceki ölçümlerle harita öğrenilir. t>=2 için LiDAR konum
ve harita düzeltmeleri, hata etiketi veya NIS kararı gözetmeden, dört koşulda
da durdurulur. Encoder/gyro tahmini devam eder. Bu planlı kontrol deneyidir;
otomatik hata tespiti veya hareket seçimi değildir.
"""

import argparse
import csv
from datetime import datetime, timezone
from itertools import groupby
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp_24_motion_comparison import BRANCH_TIME, MOTION_RATES, check_scenarios, generate_scenario
from storage.files import write_json, write_jsonl
from experiments.line_slam_common import replay, snapshot
from evaluation.trajectory import evaluate
from storage.files import write_csv
from estimation.mounting_hypotheses import compare_mounting_hypotheses
from perception.line_association import associate_lines, transform_line
from perception.line_features import extract_lines
from simulator.encoder import EncoderMeasurement
from simulator.imu import GyroMeasurement
from simulator.mounting import SensorMount


def run_frozen_window(input_path, output):
    """Yalnız ölçüm kaydını alır; gerçek konum veya hata etiketi almaz."""
    events = [json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not events:
        raise ValueError('Boş ölçüm kaydı.')
    previous, seen = -math.inf, set()
    for event in events:
        time_s = float(event['time_s'])
        key = (event['sensor_type'], event['sensor_id'], time_s)
        if (
            event['schema_version'] != 1
            or not math.isfinite(time_s)
            or time_s < 0 or time_s < previous or key in seen
        ):
            raise ValueError('Geçersiz veya tekrarlanan ölçüm paketi.')
        seen.add(key)
        previous = time_s

    warmup = [event for event in events if event['time_s'] < BRANCH_TIME]
    suffix = [event for event in events if event['time_s'] >= BRANCH_TIME]
    if not warmup or not suffix:
        raise ValueError('Kayıt hem öğrenme hem teşhis penceresini kapsamalı.')
    warmup_path = output / 'warmup_measurements.jsonl'
    write_jsonl(warmup_path, warmup)
    paths, slam, _, _, _ = replay(warmup_path, use_gps=False)
    path = list(paths['line_slam'])
    map_state_before = slam.state[4:].copy()
    map_covariance_before = slam.P[4:, 4:].copy()
    map_count_before = slam.landmark_count
    line_updates_before = slam.line_updates
    if map_count_before == 0:
        raise ValueError('Öğrenme penceresinde harita oluşmadı.')

    mounts = {'front': SensorMount(x=.25), 'rear': SensorMount(x=-.25, yaw=math.pi)}
    hypotheses, scan_rows, innovation_rows = [], [], []
    for time_s, grouped in groupby(suffix, key=lambda event: float(event['time_s'])):
        packets = list(grouped)
        encoders = [e for e in packets if e['sensor_type'] == 'encoder']
        gyros = [e for e in packets if e['sensor_type'] == 'gyro']
        if encoders or gyros:
            if len(encoders) != 1 or len(gyros) != 1:
                raise ValueError('Eşlenemeyen encoder/gyro aralığı.')
            slam.predict(
                EncoderMeasurement(time_s=time_s, **encoders[0]['data']),
                GyroMeasurement(time_s=time_s, **gyros[0]['data']),
            )
        if not math.isclose(slam.time_s, time_s, rel_tol=0., abs_tol=1e-9):
            raise ValueError('Ölçüm zamanı kestiriciyle uyuşmuyor.')

        references = slam.map_features()
        observations = []
        lidar_packets = sorted(
            (e for e in packets if e['sensor_type'] == 'lidar'),
            key=lambda e: e['sensor_id'],
        )
        for packet in lidar_packets:
            sensor = packet['sensor_id']
            if sensor not in mounts:
                raise ValueError('Tanımsız LiDAR montajı.')
            mount = mounts[sensor]
            features = extract_lines(packet['data']['scan']['points'])
            world = [transform_line(f, mount.world_pose(slam.pose)) for f in features]
            matches = associate_lines(world, references)
            counts = {status: 0 for status in ('matched', 'no_candidate', 'ambiguous', 'conflict')}
            nis_values = []
            for match in matches:
                counts[match['status']] += 1
                innovation = None
                if match['status'] == 'matched':
                    feature = features[match['observation_index']]
                    innovation = slam.line_innovation(match['reference_index'], feature, mount)
                    nis_values.append(innovation['nis'])
                    observations.append((sensor, match['reference_index'], feature, mount))
                innovation_rows.append({
                    'time_s': time_s, 'sensor_id': sensor,
                    'observation_index': match['observation_index'],
                    'status': match['status'],
                    'reference_index': match['reference_index'],
                    'innovation': innovation,
                })
            scan_rows.append({
                'time_s': time_s, 'sensor_id': sensor,
                'feature_count': len(features), **counts,
                'mean_nis': sum(nis_values) / len(nis_values) if nis_values else None,
            })
        if lidar_packets:
            hypotheses.append({
                'time_s': time_s,
                **compare_mounting_hypotheses(slam, observations),
            })
        path.append(snapshot(slam, time_s))

    checks = {
        'landmark_count_frozen': slam.landmark_count == map_count_before,
        'map_state_frozen': bool(np.array_equal(slam.state[4:], map_state_before)),
        'map_covariance_frozen': bool(np.allclose(slam.P[4:, 4:], map_covariance_before, rtol=0., atol=1e-12)),
        'no_lidar_corrections_in_window': slam.line_updates == line_updates_before,
        'prediction_reaches_record_end': math.isclose(slam.time_s, events[-1]['time_s'], abs_tol=1e-9),
    }
    if not all(checks.values()):
        raise RuntimeError(f'Sabit referans kontrolleri başarısız: {checks}')
    return {
        'path': path, 'slam': slam,
        'hypotheses': hypotheses,
        'scan_rows': scan_rows,
        'innovation_rows': innovation_rows,
        'checks': checks,
    }


def metrics_for_case(name, evaluated, hypotheses):
    estimates = [row for row in evaluated['line_slam'] if row['time_s'] > BRANCH_TIME]
    frames = [row for row in hypotheses if row['time_s'] > BRANCH_TIME]
    supported = [row for row in frames if row['status'] == 'evaluated']
    margins = []
    offsets = []
    for frame in supported:
        models = {model['hypothesis']: model for model in frame['models']}
        front = models['front_yaw']
        margins.append(front['nll_improvement_over_nominal'] - max(
            models[other]['nll_improvement_over_nominal']
            for other in ('rear_yaw', 'heading_offset')
        ))
        offsets.append(front['best_offset_deg'])
    return {
        'case': name,
        'post_branch_position_rmse_m': math.sqrt(sum(row['position_error_m']**2 for row in estimates) / len(estimates)),
        'post_branch_heading_rmse_rad': math.sqrt(sum(row['heading_error_rad']**2 for row in estimates) / len(estimates)),
        'post_branch_scan_instants': len(frames),
        'hypothesis_evaluated_instants': len(supported),
        'insufficient_support_instants': len(frames) - len(supported),
        'mean_front_fit_margin': sum(margins) / len(margins) if margins else None,
        'mean_front_offset_deg': sum(offsets) / len(offsets) if offsets else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fault-yaw-deg', type=float, default=2.)
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.fault_yaw_deg) or not 0 < abs(args.fault_yaw_deg) <= 10:
        parser.error('Hata açısının mutlak değeri 0 ile 10° arasında olmalı; sıfır hariç.')

    cases = {
        f'{condition}_{motion}': generate_scenario(motion, yaw)
        for condition, yaw in (('healthy', 0.), ('front_fault', args.fault_yaw_deg))
        for motion in MOTION_RATES
    }
    checks = check_scenarios(cases)
    if not all(checks.values()):
        raise RuntimeError('Eşleştirilmiş kayıt üretimi başarısız.')
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'results' / f'exp_25_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print('EXP-25: t>=2 için dört koşulda da LiDAR düzeltmeleri kapalı; GPS güncellemesi kapalı.')

    rows = []
    for name, case in cases.items():
        folder = output / name
        folder.mkdir()
        input_path = folder / 'measurements.jsonl'
        truth_path = folder / 'ground_truth.csv'
        write_jsonl(input_path, case['events'])
        write_csv(truth_path, case['truth'])
        write_json(folder / 'simulation_metadata.json', case['metadata'])
        source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        reference_summary = None
        if args.reference_run is not None:
            reference_folder = args.reference_run / name
            reference_input = reference_folder / 'measurements.jsonl'
            if hashlib.sha256(reference_input.read_bytes()).hexdigest() != source_hash:
                raise ValueError(f'EXP-24 referansı aynı ölçümleri içermiyor: {name}')
            reference_summary = json.loads((reference_folder / 'summary.json').read_text(encoding='utf-8'))
        result = run_frozen_window(input_path, folder)
        # Gerçek konum ancak bütün kestirim ve hipotezler üretildikten sonra okunur.
        _, evaluated, _ = evaluate({'line_slam': result['path']}, truth_path)
        metrics = metrics_for_case(name, evaluated, result['hypotheses'])
        write_csv(folder / 'line_slam.csv', evaluated['line_slam'])
        write_csv(folder / 'lidar_scan_diagnostics.csv', result['scan_rows'])
        write_jsonl(folder / 'lidar_innovations.jsonl', result['innovation_rows'])
        write_jsonl(folder / 'mounting_hypotheses.jsonl', result['hypotheses'])
        child_checks = {**result['checks'], 'input_unchanged': source_hash == hashlib.sha256(input_path.read_bytes()).hexdigest()}
        if not all(child_checks.values()):
            raise RuntimeError('Kaynak veri veya sabit referans değişti.')
        for key, passed in child_checks.items():
            checks[f'{name}_{key}'] = passed
        comparison = dict(metrics)
        if reference_summary is not None:
            comparison.update({
                'exp24_position_rmse_m': reference_summary['post_branch_position_rmse_m'],
                'exp24_hypothesis_support': reference_summary['hypothesis_evaluated_instants'],
                'exp24_front_fit_margin': reference_summary['mean_front_fit_margin'],
                'exp24_hold_trigger_s': reference_summary['hold_trigger_s'],
            })
        rows.append(comparison)
        write_json(folder / 'summary.json', {
            **comparison, 'checks': child_checks,
            'input_sha256': source_hash,
            'window_start_s': BRANCH_TIME,
            'window_policy': 'LiDAR pose and map corrections disabled on a fixed schedule',
        })
        margin = metrics['mean_front_fit_margin']
        text = f'{margin:.3f}' if margin is not None else 'yetersiz destek'
        print(
            f"{name}: konum RMSE={metrics['post_branch_position_rmse_m']:.6f} m | "
            f"destek={metrics['hypothesis_evaluated_instants']}/{metrics['post_branch_scan_instants']} | "
            f"ön hipotez farkı={text}"
        )
        if reference_summary is not None:
            print(f"  EXP-24 kilit başlangıcı: {reference_summary['hold_trigger_s']}")

    write_csv(output / 'comparison.csv', rows)
    write_json(output / 'summary.json', {
        'experiment_id': 'exp_25', 'checks': checks, 'all_passed': all(checks.values()),
        'fault_yaw_deg_for_offline_evaluation': args.fault_yaw_deg,
        'reference_run': str(args.reference_run) if args.reference_run else None,
        'cases': rows,
        'limitations': [
            'Planlı teşhis penceresi tüm koşullarda aynıdır; otomatik tetikleme veya hareket seçimi yoktur.',
            'Sabit tutulan yalnız harita değildir: LiDAR ile konum düzeltmeleri de askıya alınır.',
            'Robotun hareket tahmini ve robot kovaryansı gelişmeye devam eder.',
            'İlk hatalı tarama hareketler için ortaktır; özet metrikler t>2 aralığındadır.',
            'Hipotezler nominal eşleşmelere koşulludur; eşleştirme yanlılığı sürer.',
            'NLL farkları kalibre edilmiş teşhis olasılığı veya başarı oranı değildir.',
            'Tek ortam ve tek gürültü tohumu; farklı destek sayıları puan karşılaştırmasını etkiler.',
        ],
    })
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [row['case'].replace('front_fault', 'fault').replace('_', '\n') for row in rows]
    axes[0].bar(labels, [row['post_branch_position_rmse_m'] for row in rows], color=['steelblue']*2 + ['darkorange']*2)
    axes[0].set(ylabel='Konum RMSE (m)', title='Sabit referans: hareket sonrası konum hatası')
    axes[1].bar(labels, [row['hypothesis_evaluated_instants'] for row in rows], color=['steelblue']*2 + ['darkorange']*2)
    axes[1].set(ylabel='Karşılaştırılabilen an sayısı', title='Hipotez karşılaştırma desteği', ylim=(0, 4.5))
    axes[1].set_yticks(range(5))
    figure.tight_layout()
    figure.savefig(output / 'comparison.png', dpi=160)
    plt.close(figure)
    print('Sabit referans ve kayıt kontrolleri: PASS')
    print(f'Sonuç klasörü: {output}')


if __name__ == '__main__':
    main()
