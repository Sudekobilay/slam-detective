"""Tamamlanmış EXP-40 kaydının doğrulanmış hareketli gösterimi.

Denetleyici sadece ölçümleri yeniden işler. Gerçek poz, kusursuz ortam
ve gerçek hata ayarları yalnız bu yeniden işlemenin ardından çizime girer.
"""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from itertools import groupby
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from estimation.active_pipeline import geometry_view
from estimation.diagnostic_motion import CANDIDATES
from estimation.scan_alignment_guard import ScanGuardConfig
from storage.files import write_json
from configuration.layouts import read_layout
from application.layout_service import create_agent
from simulator.environment import Environment, Rectangle
from simulator.robot import Pose, wrap_angle


FILES = ('summary.json', 'used_config.json', 'calibration.json',
         'simulation_metadata.json', 'measurements.jsonl', 'ground_truth.csv',
         'reference_evaluation.csv', 'output_evaluation.csv', 'agent_history.csv',
         'commands.csv', 'decision.json')


def hashes(folder):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in FILES}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def indexed_rows(path):
    rows = csv_rows(path)
    result = {round(float(row['time_s']), 9): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Tekrarlanan zaman: {path.name}')
    return result


def collect_frames(folder, tolerance=1e-6):
    """Gerçek konumu okumadan kestirimi yeniden oynatır ve kayda karşı sınar."""
    folder = Path(folder)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Pozitif doğrulama toleransı gerekli.')
    before = hashes(folder)
    summary = load_json(folder / 'summary.json')
    if summary.get('experiment_id') != 'exp_40' or not summary.get('required_horizon_completed'):
        raise ValueError('Tamamlanmış bir EXP-40 klasörü gerekli.')
    data = load_json(folder / 'used_config.json')
    calibration = load_json(folder / 'calibration.json')
    metadata = load_json(folder / 'simulation_metadata.json')
    if calibration['layout'] != data or metadata['layout'] != data:
        raise ValueError('Kayıt ve kalibrasyon yerleşimleri uyuşmuyor.')
    config = ScanGuardConfig(**calibration['config'])
    commands = csv_rows(folder / 'commands.csv')
    if not commands:
        raise ValueError('Komut kaydı boş.')
    rates = (float(commands[0]['left_rate_rad_s']), float(commands[0]['right_rate_rad_s']))
    normal = next((name for name, value in CANDIDATES.items() if value == rates), None)
    if normal is None:
        raise ValueError('Başlangıç komutu desteklenen normal hareketlerden değil.')
    agent = create_agent(
        data, config, summary['algorithm'], policy='passive', normal=normal,
        particles=calibration['particles'], seed=metadata['world_config']['gyro_seed'] + 20000,
        use_gps=data['gps_enabled'],
    )
    # Hareketler yeniden seçilmez: ölçüm kaydı zaten uygulanmış komutların sonucudur.
    # Pasif politika yalnız kestirim/izleyici tekrarı içindir; ekrandaki mod/seçim
    # kaynak kaydın geçmişinden alınır. Hiçbir yeni deney sonucu üretilmez.
    saved_reference = indexed_rows(folder / 'reference_evaluation.csv')
    saved_output = indexed_rows(folder / 'output_evaluation.csv')
    history = indexed_rows(folder / 'agent_history.csv')
    events = [json.loads(line) for line in (folder / 'measurements.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    frames, scans, maximum_error = [], {}, 0.
    for time, batch in groupby(events, key=lambda row: row['time_s']):
        batch = list(batch)
        agent.process(batch)
        key = round(float(time), 9)
        for expected_rows, actual in ((saved_reference, agent.reference_path[-1]),
                                      (saved_output, agent.path[-1])):
            if key not in expected_rows:
                raise ValueError('Kayıtlı kestirim zamanı eksik.')
            expected = expected_rows[key]
            errors = [abs(actual[name] - float(expected[name])) for name in ('x_m', 'y_m')]
            errors.append(abs(wrap_angle(actual['theta_rad'] - float(expected['theta_rad']))))
            maximum_error = max(maximum_error, *errors)
        saved = history[key]
        if (saved['map_quarantined'] == 'True') != agent.guard.quarantine_active:
            raise ValueError('Harita koruması yeniden oynatmada uyuşmuyor.')
        if saved['joint_status'] != agent.history[-1]['joint_status']:
            raise ValueError('Alarm geçmişi yeniden oynatmada uyuşmuyor.')
        for event in batch:
            if event['sensor_type'] == 'lidar':
                scans[event['sensor_id']] = {'time_s': time, 'points': event['data']['scan']['points']}
        frames.append({
            'time_s': time, 'reference': dict(agent.reference_path[-1]),
            'output': dict(agent.path[-1]), 'history': saved,
            'scans': dict(scans),
            'map': [(feature.start, feature.end) for feature in geometry_view(agent.reference).map_features()],
            'geometry_alarm': agent.guard.confirmed, 'gps_alarm': agent.gps_monitor.confirmed,
        })
    keys = {round(float(f['time_s']), 9) for f in frames}
    if keys != set(saved_reference) or keys != set(saved_output) or keys != set(history):
        raise ValueError('Ölçüm ve kestirim zaman kapsamları uyuşmuyor.')
    if maximum_error > tolerance:
        raise ValueError(f'Yeniden oynatma uyuşmuyor: en büyük poz/yön farkı {maximum_error:.3e}.')
    if agent.trigger_time_s != summary['trigger_time_s']:
        raise ValueError('Alarm zamanı uyuşmuyor.')
    # Gerçek bilgiler yalnız doğrulanan kestirimlerden sonra çizim için okunur.
    truth = indexed_rows(folder / 'ground_truth.csv')
    if keys != set(truth):
        raise ValueError('Gerçek konumun çizim zamanları uyuşmuyor.')
    for frame in frames:
        frame['truth'] = {name: float(value) for name, value in truth[round(frame['time_s'], 9)].items()}
    return frames, data, metadata, summary, load_json(folder / 'decision.json'), before, maximum_error


def display_decision(time_s, summary, decision):
    trigger = summary['trigger_time_s']
    return decision if trigger is not None and time_s >= trigger - 1e-9 else None


def physical_scan(sensor, scan, truth, mount, fault):
    """Yalnız çizim: ışınları taramanın alındığı gerçek pozda gösterir."""
    if fault['type'] in ('lidar', 'combined') and fault['sensor'] == sensor and scan['time_s'] >= fault['start_s']:
        mount = replace(mount, yaw=mount.yaw + math.radians(fault['yaw_deg']))
    pose = Pose(truth['x_m'], truth['y_m'], truth['theta_rad'])
    origin = mount.world_pose(pose)
    points = [mount.point_to_world(point, pose) for point in scan['points'] if point is not None]
    return (origin.x, origin.y), points


def render_demo(folder, output, *, fps=6, tolerance=1e-6):
    if type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError('FPS 1–30 arasında olmalı.')
    frames, data, metadata, summary, decision, before, error = collect_frames(folder, tolerance)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    units = read_layout(data)
    truth_by_time = {round(frame['time_s'], 9): frame['truth'] for frame in frames}
    environment = Environment(Rectangle(**metadata['environment']['boundary']),
                              tuple(Rectangle(**row) for row in metadata['environment']['obstacles']))
    walls = [(wall.start, wall.end) for wall in environment.walls]
    colors = {unit.sensor_id: plt.get_cmap('tab10')(index % 10) for index, unit in enumerate(units)}
    bounds = (min(f['truth']['x_m'] for f in frames) - .7,
              max(f['truth']['x_m'] for f in frames) + .7,
              min(f['truth']['y_m'] for f in frames) - .7,
              max(f['truth']['y_m'] for f in frames) + .7)
    figure = plt.figure(figsize=(15, 6))
    figure.subplots_adjust(left=.05, right=.98, bottom=.17, top=.80, wspace=.30, hspace=.65)
    grid = figure.add_gridspec(2, 3, width_ratios=[1., 1.1, 1.05])
    global_axis = figure.add_subplot(grid[:, 0])
    local_axis = figure.add_subplot(grid[:, 1])
    status_axis = figure.add_subplot(grid[0, 2])
    score_axis = figure.add_subplot(grid[1, 2])
    modes = {'NORMAL': 'Normal sürüş', 'WATCH': 'Tutarsızlık inceleniyor',
             'DIAGNOSING': 'Teşhis hareketi', 'PASSIVE_OBSERVATION': 'Normal rotada gözlem',
             'NO_SUPPORTED_MANEUVER': 'Desteklenen hareket yok',
             'DIAGNOSIS_WINDOW_ENDED': 'Teşhis penceresi tamamlandı'}
    labels = {'straight': 'Düz', 'arc': 'Yay'}

    def draw(index):
        frame, prefix = frames[index], frames[:index + 1]
        for axis in (global_axis, local_axis):
            axis.clear()
            axis.set_aspect('equal', adjustable='box')
            axis.set(xlabel='x (m)', ylabel='y (m)')
            axis.grid(alpha=.2)
            axis.add_collection(LineCollection(walls, colors='black', linewidths=1.))
            if frame['map']:
                axis.add_collection(LineCollection(frame['map'], colors='mediumpurple', linewidths=2, label='Öğrenilen referans haritası'))
            axis.plot([f['truth']['x_m'] for f in prefix], [f['truth']['y_m'] for f in prefix], color='darkorange', label='Gerçek rota')
            axis.plot([f['reference']['x_m'] for f in prefix], [f['reference']['y_m'] for f in prefix], '--', color='royalblue', label='GPS’siz referans')
            if data['gps_enabled']:
                axis.plot([f['output']['x_m'] for f in prefix], [f['output']['y_m'] for f in prefix], ':', color='forestgreen', label='GPS füzyon çıktısı')
        boundary = metadata['environment']['boundary']
        global_axis.set(xlim=(boundary['x_min'] - .3, boundary['x_max'] + .3),
                        ylim=(boundary['y_min'] - .3, boundary['y_max'] + .3), title='Ortam ve öğrenilen harita')
        global_axis.legend(fontsize=7, loc='upper right')
        local_axis.set(xlim=bounds[:2], ylim=bounds[2:], title='Robot ve son LiDAR taramaları')
        truth = frame['truth']
        local_axis.add_patch(plt.Circle((truth['x_m'], truth['y_m']), .35,
                                       fill=False, color='darkorange', lw=2))
        local_axis.arrow(truth['x_m'], truth['y_m'], .45*math.cos(truth['theta_rad']),
                         .45*math.sin(truth['theta_rad']), head_width=.07, color='darkorange', length_includes_head=True)
        for unit in units:
            sensor = unit.sensor_id
            if sensor not in frame['scans']:
                continue
            scan = frame['scans'][sensor]
            old_truth = truth_by_time[round(scan['time_s'], 9)]
            origin, points = physical_scan(sensor, scan, old_truth, unit.actual_mount, metadata['layout_fault'])
            if points:
                local_axis.add_collection(LineCollection([(origin, point) for point in points[::8]], colors=[colors[sensor]], linewidths=.6, alpha=.25))
                local_axis.scatter(*zip(*points), s=5, color=colors[sensor], alpha=.4)
            local_axis.scatter(*origin, s=28, marker='s', color=colors[sensor], label=sensor)
        local_axis.legend(fontsize=7, loc='lower left', ncol=2)
        status_axis.clear()
        status_axis.axis('off')
        h = frame['history']
        ages = [frame['time_s'] - s['time_s'] for s in frame['scans'].values()]
        mode = modes.get(h['mode'], h['mode'])
        motion = labels.get(h['selected_motion'], 'Normal rota / seçim yok')
        status_axis.set_title('Kayıtlı karar ve sistem durumu', fontsize=11)
        text = (f"Zaman: {frame['time_s']:.1f} s\nDurum: {mode}\nSeçilen hareket: {motion}\n"
                f"Geometrik alarm: {'VAR' if frame['geometry_alarm'] else 'YOK'}\n"
                f"GPS alarmı: {'VAR' if frame['gps_alarm'] else 'YOK'}\n"
                f"GPS füzyonu: {'Açık' if data['gps_enabled'] else 'Kapalı'}\n"
                f"Harita koruması: {'Aktif' if h['map_quarantined'] == 'True' else 'Pasif'}\n"
                f"Son taramanın yaşı: {max(ages, default=0.):.1f} s")
        status_axis.text(.02, .95, text, va='top', fontsize=10, linespacing=1.5)
        score_axis.clear()
        visible = display_decision(frame['time_s'], summary, decision)
        candidates = visible.get('candidates', []) if visible else []
        supported = [row for row in candidates if row.get('score_m2') is not None]
        if supported:
            score_axis.barh([labels[row['motion']] for row in supported], [row['score_m2'] for row in supported],
                            color=['royalblue' if row['motion'] == summary['selected_motion'] else 'lightgray' for row in supported])
            score_axis.set_xlabel('Beklenen geometrik ayrım (m²)', fontsize=9)
            score_axis.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
            score_axis.set_title('Alarm anındaki aday puanları', fontsize=10)
            score_axis.text(0., -.52, 'Mavi: seçilen aday. Puan, teşhis olasılığı değildir.\nUygunluk/harita açıklığı da seçimde değerlendirilir.',
                            transform=score_axis.transAxes, fontsize=8)
        else:
            score_axis.axis('off')
            score_axis.text(.02, .65, 'Henüz aday puanı yok.' if visible is None else 'Bu kayıtta destekli aday puanı yok.', fontsize=10)
        figure.suptitle(f"SLAM-Detective | EXP-40 | {data['name']} | {len(units)} LiDAR | {summary['algorithm']}\n"
                       'Kayıt yeniden gösterimi; gerçek konum ve fiziksel ışınlar yalnız çizimdedir.', fontsize=12)

    animation = FuncAnimation(figure, draw, frames=len(frames), interval=1000/fps, blit=False)
    print(f'GIF hazırlanıyor: {len(frames)} kare...', flush=True)
    animation.save(output / 'active_demo.gif', writer=PillowWriter(fps=fps), dpi=100)
    draw(len(frames) - 1)
    figure.savefig(output / 'final_frame.png', dpi=140)
    plt.close(figure)
    unchanged = hashes(Path(folder)) == before
    write_json(output / 'summary.json', {
        'source_run': str(folder), 'source_hashes': before,
        'frame_count': len(frames), 'fps': fps, 'algorithm': summary['algorithm'],
        'layout': data['name'], 'lidar_count': len(units),
        'replayed_estimates_match_saved': True, 'maximum_pose_component_error': error,
        'source_files_unchanged': unchanged,
        'limitations': [
            'Yeni deney veya hareket seçimi sonucu değildir; kaydedilmiş seçim ve komutların oluşturduğu ölçümler gösterilir.',
            'Aday puanları yalnız kayıtlı alarm anından sonra gösterilir; gerçek hata etiketi teşhis sonucu diye sunulmaz.',
            'Işınlar son tarama anındaki fiziksel pozda tutulur; ara karelerde yeni tarama üretilmez.',
            'Çizgi haritası GPS almayan referanstır; varsa GPS füzyon çıktısı ayrı yörüngedir.',
            'Parçacıklı yeniden oynatma farklı sayısal ortamlarda tolerans dışına çıkarsa gösterim durdurulur.',
        ],
    })
    if not unchanged:
        raise RuntimeError('Kaynak kayıt değişti.')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = render_demo(args.run, root / 'results' / f'active_demo_{stamp}', fps=args.fps, tolerance=args.tolerance)
    print('Kayıtlı kestirim ve alarm doğrulaması: PASS')
    print('Kaynak dosyaları korunmuş: PASS')
    print(f'Hareketli gösterim: {output / "active_demo.gif"}')
    print(f'Son kare: {output / "final_frame.png"}')


if __name__ == '__main__':
    main()
