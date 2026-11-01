"""EXP-26: Yalnız geçmiş ölçümlerden geometrik teşhis hareketi seçimi.

Karar t=2'deki kestirim ve öğrenilmiş haritadan verilir. Seçilen hareketin
kaydı sonradan yeniden üretilir ve ortak geçmişinin aynı olduğu doğrulanır.
Bu iki aşamalı simülasyon deneyidir; kesintisiz canlı kontrol döngüsü değildir.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from estimation.diagnostic_motion import select_diagnostic_motion
from experiments.exp_24_motion_comparison import (
    BRANCH_TIME, generate_scenario, write_json, write_jsonl,
)
from experiments.exp_25_frozen_reference import run_frozen_window, metrics_for_case
from experiments.line_slam_common import evaluate, write_csv


def select_from_prefix(input_path, output):
    """Gelecek ölçüm veya gerçek dünya bilgisi almayan karar sınırı."""
    events = [json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not events or max(event['time_s'] for event in events) != BRANCH_TIME:
        raise ValueError('Karar kaydı tam t=2 anında bitmeli; gelecek ölçüm içermemeli.')
    if any(event['time_s'] > BRANCH_TIME for event in events):
        raise ValueError('Karar girdisinde gelecek ölçüm var.')
    inference = run_frozen_window(input_path, output)
    slam = inference['slam']
    decision = select_diagnostic_motion(slam)
    # Bu dosya yalnız tahminleri içerir; gerçek harita veya hata etiketi içermez.
    belief = {
        'time_s': slam.time_s,
        'state': slam.state.tolist(),
        'covariance': slam.P.tolist(),
        'map_lines': [
            {'start': list(feature.start), 'end': list(feature.end)}
            for feature in slam.map_features()
        ],
    }
    write_json(output / 'planning_belief.json', belief)
    write_json(output / 'decision.json', decision)
    return decision, belief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fault-yaw-deg', type=float, default=2.)
    args = parser.parse_args()
    if not math.isfinite(args.fault_yaw_deg) or not 0 < abs(args.fault_yaw_deg) <= 10:
        parser.error('Hata açısının mutlak değeri 0 ile 10° arasında olmalı; sıfır hariç.')
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'results' / f'exp_26_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print('EXP-26: Geometrik hareket seçimi; GPS kapalı, t>=2 LiDAR düzeltmeleri kapalı.')
    comparisons, decisions, beliefs, checks = [], {}, {}, {}
    for condition, yaw in (('healthy', 0.), ('front_fault', args.fault_yaw_deg)):
        condition_dir = output / condition
        condition_dir.mkdir()
        # Kayıt üreticisi deney koşulunu bilir. Karar fonksiyonuna yalnız
        # t<=2 ölçümleri geçer; üreticinin geri kalanı ve metadata geçmez.
        template = generate_scenario('arc', yaw)
        prefix = [event for event in template['events'] if event['time_s'] <= BRANCH_TIME]
        prefix_dir = condition_dir / 'decision_prefix'
        prefix_dir.mkdir()
        prefix_path = prefix_dir / 'measurements.jsonl'
        write_jsonl(prefix_path, prefix)
        decision, belief = select_from_prefix(prefix_path, prefix_dir)
        decisions[condition], beliefs[condition] = decision, belief
        choice = decision['selected_motion']
        print(f'{condition}: seçilen hareket={choice} | durum={decision["status"]}')
        for candidate in decision['candidates']:
            print(
                f"  {candidate['motion']}: beklenen ayrım={candidate['predicted_separation_score']} | "
                f"uygun={candidate['eligible']}"
            )
        if choice is None:
            checks[f'{condition}_supported_choice'] = False
            # Desteksiz bir kararı zorla düz veya dönüşe çevirmiyoruz.
            continue
        checks[f'{condition}_supported_choice'] = True
        row_by_motion = {}
        hashes = {}
        for strategy, motion in (('fixed_straight', 'straight'), ('fixed_arc', 'arc'), ('selected', choice)):
            case = generate_scenario(motion, yaw)
            future_prefix = [event for event in case['events'] if event['time_s'] <= BRANCH_TIME]
            same_prefix = future_prefix == prefix
            checks[f'{condition}_{strategy}_same_prefix'] = same_prefix
            if not same_prefix:
                raise RuntimeError('Seçim sonrası kaydın ortak geçmişi değişti.')
            folder = condition_dir / strategy
            folder.mkdir()
            source = folder / 'measurements.jsonl'
            truth = folder / 'ground_truth.csv'
            write_jsonl(source, case['events'])
            write_csv(truth, case['truth'])
            write_json(folder / 'simulation_metadata.json', case['metadata'])
            initial_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            # Gerçek ortamın çarpışma kontrolü generate_scenario içindedir.
            # Kestirici ve hareket seçici bu gerçek haritayı almaz.
            result = run_frozen_window(source, folder)
            _, evaluated, _ = evaluate({'line_slam': result['path']}, truth)
            metrics = metrics_for_case(f'{condition}_{strategy}', evaluated, result['hypotheses'])
            metrics.update({'condition': condition, 'strategy': strategy, 'executed_motion': motion})
            comparisons.append(metrics)
            write_csv(folder / 'line_slam.csv', evaluated['line_slam'])
            write_csv(folder / 'lidar_scan_diagnostics.csv', result['scan_rows'])
            write_jsonl(folder / 'mounting_hypotheses.jsonl', result['hypotheses'])
            write_json(folder / 'summary.json', {'metrics': metrics, 'checks': result['checks'], 'input_sha256': initial_hash})
            checks[f'{condition}_{strategy}_input_unchanged'] = initial_hash == hashlib.sha256(source.read_bytes()).hexdigest()
            hashes[strategy] = initial_hash
            if strategy != 'selected':
                row_by_motion[motion] = metrics
            else:
                checks[f'{condition}_selected_equals_corresponding_fixed_record'] = initial_hash == hashes[f'fixed_{motion}']
                checks[f'{condition}_selected_equals_corresponding_fixed_error'] = metrics['post_branch_position_rmse_m'] == row_by_motion[motion]['post_branch_position_rmse_m']
            margin = metrics['mean_front_fit_margin']
            margin_text = f'{margin:.3f}' if margin is not None else 'yetersiz destek'
            print(
                f"  {strategy}/{motion}: konum RMSE={metrics['post_branch_position_rmse_m']:.6f} m | "
                f"destek={metrics['hypothesis_evaluated_instants']}/{metrics['post_branch_scan_instants']} | "
                f"ön hipotez farkı={margin_text}"
            )
        if not all(value for key, value in checks.items() if key.startswith(condition)):
            raise RuntimeError('Seçim veya kayıt tutarlılık kontrolü başarısız.')

    write_csv(output / 'comparison.csv', comparisons)
    write_json(output / 'summary.json', {
        'experiment_id': 'exp_26', 'checks': checks,
        'all_supported': all(decision['selected_motion'] is not None for decision in decisions.values()),
        'decisions': decisions, 'cases': comparisons,
        'same_planning_belief_across_sensor_conditions': beliefs.get('healthy') == beliefs.get('front_fault'),
        'limitations': [
            'Karar zamanı önceden t=2 olarak belirlenir; otomatik anomali tetiklemesi yoktur.',
            'Seçici geometrik ayırt ediciliği kullanır; mevcut hata etiketi veya gelecek ölçüm kullanmaz.',
            'Bu tasarımda sağlıklı ve hatalı koşulda planlama inancı aynı olabilir ve aynı hareket seçilebilir.',
            'Görünürlük nominal FOV ve öğrenilmiş sonlu çizgi desteklerinden yaklaşık hesaplanır.',
            'Ham taramadan gelecekteki özellik çıkarımı ve eşleştirme başarısı garanti edilmez.',
            'Skor küçük açısal hata duyarlılıklarına dayanır; 1 derece yalnız ortak karşılaştırma ölçeğidir.',
            'Farklı gelecek taramaların korelasyonları skor birikiminde modellenmez; skor gerçek bilgi kazanımı değildir.',
            'Bilinen harita parçalarına uzaklık kontrolü bilinmeyen alan için güvenlik garantisi değildir.',
            'Gerçek simülatör hareket sırasında ayrı çarpışma kontrolü uygular.',
            'GPS sapması hareket seçicinin hipotezleri arasında henüz yoktur.',
            'Tek ortam ve tohum; seçilen hareketin daha iyi çıkacağı varsayılmaz.',
            'Seçilen hareket sabit adaylardan biri olduğundan o adayın kaydıyla aynı sonucu vermelidir.',
        ],
    })
    if comparisons:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for condition, color in (('healthy', 'steelblue'), ('front_fault', 'darkorange')):
            subset = [row for row in comparisons if row['condition'] == condition]
            xs = np.arange(len(subset)) + (-.18 if condition == 'healthy' else .18)
            axes[0].bar(xs, [row['post_branch_position_rmse_m'] for row in subset], width=.35, label=condition, color=color)
            axes[1].bar(xs, [row['hypothesis_evaluated_instants'] for row in subset], width=.35, label=condition, color=color)
        for axis in axes:
            axis.set_xticks(range(3), ['Sabit düz', 'Sabit dönüş', 'Seçilen'])
            axis.legend(fontsize=8)
        axes[0].set(title='Hareket sonrası konum hatası', ylabel='RMSE (m)')
        axes[1].set(title='Hipotez karşılaştırma desteği', ylabel='Tarama anı sayısı', ylim=(0,4.5))
        figure.tight_layout()
        figure.savefig(output / 'comparison.png', dpi=160)
        plt.close(figure)
    print(f'Sonuç klasörü: {output}')


if __name__ == '__main__':
    main()
