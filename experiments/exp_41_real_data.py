"""EXP-41: Freiburg 079 gerçek FLASER/odometri kaydını yeniden oynatır."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_loaders.carmen import CarmenLoader, laser_mount, relative_pose
from estimation.active_pipeline import make_backend, geometry_view
from estimation.recorded_odometry import predict_recorded_odometry
from experiments.exp_24_motion_comparison import write_json
from experiments.line_slam_common import write_csv
from simulator.robot import Pose, wrap_angle


SOURCE_URL = 'https://www.ais.uni-bonn.de/~holz/spmicp/files/fr079.clf'
SOURCE_PAGE = 'https://www.ais.uni-bonn.de/~holz/spmicp/'
SOURCE_SHA256 = 'd7ef6348cf29c2b71702f3a3ac779cb19a28184e11a031ee4150ec3b5f4696e6'


def download_recording(path):
    path = Path(path)
    if path.suffix.lower() not in ('.clf', '.log'):
        raise ValueError('İndirilecek düz metin kaydı için .clf veya .log uzantısı kullanın.')
    if path.exists():
        raise FileExistsError(f'Dosya zaten var: {path}. --download seçeneğini kaldırın.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response, temporary.open('wb') as handle:
            total = 0
            while chunk := response.read(65536):
                total += len(chunk)
                if total > 20_000_000:
                    raise ValueError('İndirme beklenen boyutu aştı.')
                handle.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError('Kaynak dosyanın SHA-256 özeti beklenen sürümle uyuşmuyor.')
        # HTML hata sayfasını gerçek kayıt olarak kabul etme.
        iterator = iter(CarmenLoader(temporary))
        try:
            next(iterator)
        except StopIteration as error:
            raise ValueError('İndirilen dosyada FLASER yok.') from error
        finally:
            iterator.close()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def replay_recording(path, *, algorithm='ekf', max_scans=40, stride=3,
                     particles=40, seed=1101, expected_beams=360,
                     start_angle_deg=-90., angle_step_deg=.5, max_range_m=80.):
    if max_scans < 2 or stride < 1:
        raise ValueError('En az iki tarama ve pozitif seçim aralığı gerekli.')
    loader = CarmenLoader(path, expected_beams=expected_beams,
                          start_angle_deg=start_angle_deg, angle_step_deg=angle_step_deg,
                          max_range_m=max_range_m)
    backend = None
    first = previous = mount = None
    trajectory, odometry, timings, cloud = [], [], [], []
    iterator = iter(loader)
    try:
        for index, scan in enumerate(iterator):
            if index % stride:
                continue
            if first is None:
                first = scan
                mount = laser_mount(scan)
                backend = make_backend(algorithm, particles, seed, {'recorded_lidar': mount})
            current_mount = laser_mount(scan)
            if math.hypot(current_mount.x - mount.x, current_mount.y - mount.y) > .02 or abs(wrap_angle(current_mount.yaw - mount.yaw)) > .01:
                raise ValueError('Kayıttaki LiDAR/robot dönüşümü sabit değil; bu adaptör sabit montaj gerektirir.')
            time_s = scan.timestamp_s - first.timestamp_s
            started = perf_counter()
            if previous is not None:
                predict_recorded_odometry(backend, relative_pose(previous.odometry, scan.odometry), time_s)
            backend.update_scan('recorded_lidar', scan.points)
            elapsed = perf_counter() - started
            view = geometry_view(backend)
            if len(trajectory) % 5 == 0:
                sensor_pose = mount.world_pose(view.pose)
                c, s = math.cos(sensor_pose.theta), math.sin(sensor_pose.theta)
                cloud.extend((sensor_pose.x + c*p[0] - s*p[1], sensor_pose.y + s*p[0] + c*p[1])
                             for p in scan.points[::3] if p is not None)
            trajectory.append(backend.snapshot(time_s))
            relative = relative_pose(first.odometry, scan.odometry)
            odometry.append({'time_s': time_s, 'x_m': 1. + relative.x,
                             'y_m': 3. + relative.y, 'theta_rad': relative.theta})
            timings.append({'time_s': time_s, 'source_line': scan.line_number,
                            'valid_points': sum(p is not None for p in scan.points),
                            'processing_s': elapsed})
            if len(trajectory) % 10 == 0:
                print(f'Tarama: {len(trajectory)}/{max_scans}', flush=True)
            previous = scan
            if len(trajectory) >= max_scans:
                break
    finally:
        iterator.close()
    if len(trajectory) < 2:
        raise ValueError('En az iki kullanılabilir FLASER kaydı gerekli.')
    return backend, trajectory, odometry, timings, cloud, {
        'mount': asdict(mount), 'first_absolute_timestamp_s': first.timestamp_s,
        'message_counts_in_read_prefix': loader.counts,
        'selected_scans': len(trajectory), 'duration_s': trajectory[-1]['time_s'],
        'beam_profile': {'expected_beams': expected_beams, 'start_angle_deg': start_angle_deg,
                         'angle_step_deg': angle_step_deg, 'max_range_m': max_range_m},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/raw/fr079.clf'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--algorithm', choices=['ekf', 'fastslam1', 'fastslam2'], default='ekf')
    parser.add_argument('--max-scans', type=int, default=40)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--particles', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--expected-beams', type=int, default=360)
    parser.add_argument('--start-angle-deg', type=float, default=-90.)
    parser.add_argument('--angle-step-deg', type=float, default=.5)
    parser.add_argument('--max-range-m', type=float, default=80.)
    args = parser.parse_args()
    if args.download:
        print('Gerçek kayıt indiriliyor...', flush=True)
        download_recording(args.input)
    before = hashlib.sha256(args.input.read_bytes()).hexdigest()
    print('EXP-41: gerçek LiDAR + kayıtlı odometri; IMU/GPS ve aktif hareket yok.', flush=True)
    backend, trajectory, odometry, timings, cloud, metadata = replay_recording(
        args.input, algorithm=args.algorithm, max_scans=args.max_scans, stride=args.stride,
        particles=args.particles, seed=args.seed, expected_beams=args.expected_beams,
        start_angle_deg=args.start_angle_deg, angle_step_deg=args.angle_step_deg,
        max_range_m=args.max_range_m,
    )
    unchanged = hashlib.sha256(args.input.read_bytes()).hexdigest() == before
    output = Path(__file__).resolve().parents[1] / 'results' / ('exp_41_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    features = geometry_view(backend).map_features()
    write_csv(output / 'trajectory.csv', trajectory)
    write_csv(output / 'recorded_odometry.csv', odometry)
    write_csv(output / 'processing.csv', timings)
    write_json(output / 'map.json', [asdict(f) for f in features])
    summary = {
        'experiment_id': 'exp_41', 'algorithm': args.algorithm,
        'input_name': args.input.name, 'input_sha256': before,
        'source_file_unchanged': unchanged, **metadata,
        'map_line_count': len(features), 'processing_total_s': sum(r['processing_s'] for r in timings),
        'assumed_motion_noise': {'translation_density_m_sqrt_s': .02, 'rotation_density_rad_sqrt_s': .03},
        'dataset_identity': 'fr079_bonn_verified' if before == SOURCE_SHA256 else 'user_supplied_carmen',
        'source_page': SOURCE_PAGE if before == SOURCE_SHA256 else None,
        'download_url': SOURCE_URL if before == SOURCE_SHA256 else None,
        'limitations': [
            'Kayıtlı tek LiDAR ve ekli odometri; IMU/GPS veya ham encoder sayımı üretilmez.',
            'Odometri referans gerçek konum değildir; ATE/RMSE doğruluk sonucu hesaplanmaz.',
            'Kayıt üzerinde yeni hareket uygulanmaz, aktif teşhis başarısı iddia edilmez.',
            'Eski FLASER ışın açılarını taşımaz; açık 360 ışın/-90 başlangıç/0.5 derece profil varsayımı kullanılır.',
            'LiDAR/robot montajı kayıttaki iki pozdan hesaplanır; aynı fiziksel gerçeklik hakkında bağımsız doğrulama değildir.',
            'Nominal odometri gürültüsü ve mevcut eşleştirme eşikleri gerçek veri üzerinde kalibre edilmemiştir.',
            'Seçilen ön ek ve seyreltilmiş taramalar; tam veri seti benchmark sonucu değildir.',
            'Bonn sitesindeki benchmark sürümü; MIT arşivindeki gzip dosyasıyla bayt eşitliği varsayılmaz.',
        ],
    }
    write_json(output / 'summary.json', summary)
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    if cloud:
        axis.scatter(*zip(*cloud), s=1, alpha=.12, color='gray', label='Tahmini pozla yerleştirilen ölçümler')
    axis.plot([p['x_m'] for p in odometry], [p['y_m'] for p in odometry], '--', label='Kayıtlı odometri')
    axis.plot([p['x_m'] for p in trajectory], [p['y_m'] for p in trajectory], label='SLAM tahmini')
    for feature in features:
        axis.plot([feature.start[0], feature.end[0]], [feature.start[1], feature.end[1]], color='purple', alpha=.5)
    axis.set(xlabel='x (m)', ylabel='y (m)', title=f'Gerçek kayıt yeniden oynatımı — {args.algorithm}')
    axis.set_aspect('equal', adjustable='datalim')
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    figure.savefig(output / 'real_data_map.png', dpi=150)
    plt.close(figure)
    print(f"İşlenen tarama: {len(trajectory)} | kayıt süresi: {metadata['duration_s']:.2f} s")
    print(f'Harita çizgisi: {len(features)}')
    print(f"Kaynak dosyası korunmuş: {'PASS' if unchanged else 'FAIL'}")
    print('Gerçek konum yok: doğruluk RMSE/ATE hesaplanmadı.')
    print(f'Sonuç klasörü: {output}')
    if not unchanged:
        raise RuntimeError('Kaynak dosyası değişti.')


if __name__ == '__main__':
    main()
