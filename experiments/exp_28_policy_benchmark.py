"""EXP-28: İki normal rota, dört politika ve eşleştirilmiş gürültü tohumları.

Politikalar aynı ortam/sensör ayarlarıyla çalışır. Anomali öncesi ölçüm
geçmişlerinin eşitliği denetlenir. Skor veya eşik sonuca göre değiştirilmez.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from estimation.diagnostic_motion import CANDIDATES
from estimation.online_diagnostic_agent import OnlineDiagnosticAgent
from storage.files import write_json, write_jsonl
from evaluation.trajectory import evaluate
from storage.files import write_csv
from simulator.streaming_world import StreamingWorld, WorldConfig

POLICIES = ('passive', 'fixed_straight', 'fixed_arc', 'adaptive')


class FixedArcAgent(OnlineDiagnosticAgent):
    """Aynı anomali/kilit hesabıyla, alarmdan sonra sabit yay kontrolü."""
    def __init__(self):
        super().__init__(strategy='passive')

    def process(self, events):
        already_active = self.hold.active
        super().process(events)
        if not already_active and self.hold.active:
            self.selected_motion = 'arc'
            self.mode = 'DIAGNOSING'
            self.history[-1]['mode'] = self.mode

    def command(self):
        if self.hold.active and self.estimator.time_s < self.window_end_s - 1e-9:
            return CANDIDATES['arc']
        return super().command()


def make_agent(policy, normal_motion):
    if policy not in POLICIES or normal_motion not in CANDIDATES:
        raise ValueError('Geçersiz politika veya normal rota.')
    if policy == 'fixed_arc':
        agent = FixedArcAgent()
    else:
        strategy = {'passive': 'passive', 'fixed_straight': 'straight', 'adaptive': 'adaptive'}[policy]
        agent = OnlineDiagnosticAgent(strategy=strategy)
    # Bu bilinen rota komutudur; gerçek dünya/hata bilgisi değildir.
    agent.normal_command = CANDIDATES[normal_motion]
    return agent


def run_trial(policy, normal_motion, yaw, seed):
    world = StreamingWorld(WorldConfig(front_yaw_error_deg=yaw, gyro_seed=seed, gps_seed=seed+1))
    agent = make_agent(policy, normal_motion)
    events = world.initial_measurements()
    agent.process(events)
    commands, completed, stop_reason = [], True, None
    for _ in range(40):
        rates = agent.command()
        command_row = {
            'interval_start_s': agent.estimator.time_s,
            'left_rate_rad_s': rates[0], 'right_rate_rad_s': rates[1],
            'changed_from_normal': rates != agent.normal_command,
            'mode_before_step': agent.mode,
            'applied': False,
        }
        commands.append(command_row)
        try:
            batch = world.step(*rates)
        except RuntimeError as error:
            completed, stop_reason = False, str(error)
            break
        command_row['applied'] = True
        agent.process(batch)
        events.extend(batch)
    # Gerçek konum ve koşul etiketi yalnız deney yöneticisinde bulunur.
    return agent, events, commands, world.evaluation_records(), completed, stop_reason


def evidence_metrics(hypotheses, evaluation_start_s=2.):
    # Bütün politikalarda aynı değerlendirme aralığı; alarm zamanına göre
    # sonradan farklı pencereler seçilmiyor. t=2 ortak ilk hata taraması hariç.
    frames = [frame for frame in hypotheses if frame['time_s'] > evaluation_start_s]
    supported = [frame for frame in frames if frame['status'] == 'evaluated']
    margins = []
    for frame in supported:
        models = {model['hypothesis']: model for model in frame['models']}
        margins.append(models['front_yaw']['nll_improvement_over_nominal'] - max(
            models[name]['nll_improvement_over_nominal'] for name in ('rear_yaw', 'heading_offset')
        ))
    return {
        'hypothesis_frames': len(frames),
        'supported_frames': len(supported),
        'front_margin_sum': sum(margins),
        'mean_front_margin': sum(margins)/len(margins) if margins else None,
    }


def group_results(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['normal_motion'], row['policy'], row['fault_yaw_deg'])].append(row)
    summaries = []
    for (normal, policy, yaw), trials in sorted(groups.items()):
        completed = [row for row in trials if row['completed']]
        supported = sum(row['supported_frames'] for row in completed)
        frames = sum(row['hypothesis_frames'] for row in completed)
        alarms = sum(row['alarm'] for row in trials)
        delays = [row['detection_delay_s'] for row in trials if row['detection_delay_s'] is not None]
        summaries.append({
            'normal_motion': normal, 'policy': policy, 'fault_yaw_deg': yaw,
            'trials': len(trials), 'completed_trials': len(completed),
            'alarm_trials': alarms,
            'healthy_alarm_trials': alarms if yaw == 0. else 0,
            'fault_trials_without_alarm': sum(not row['alarm'] for row in trials) if yaw != 0. else 0,
            'fault_trials_with_post_fault_alarm': sum(row['detection_delay_s'] is not None for row in trials) if yaw != 0. else 0,
            'premature_alarm_trials': sum(row['premature_alarm'] for row in trials),
            'mean_detection_delay_s': sum(delays)/len(delays) if delays else None,
            'mean_full_position_rmse_m': sum(row['full_position_rmse_m'] for row in completed)/len(completed) if completed else None,
            'position_rmse_std_m': (
                math.sqrt(sum((row['full_position_rmse_m'] - sum(r['full_position_rmse_m'] for r in completed)/len(completed))**2 for row in completed)/(len(completed)-1))
                if len(completed)>1 else None
            ),
            'supported_frames': supported, 'hypothesis_frames': frames,
            'pooled_front_margin': sum(row['front_margin_sum'] for row in completed)/supported if supported else None,
            'trials_with_changed_commands': sum(row['changed_command_intervals'] > 0 for row in trials),
            'selected_straight_trials': sum(row['selected_motion'] == 'straight' for row in trials),
            'selected_arc_trials': sum(row['selected_motion'] == 'arc' for row in trials),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--fault-yaw-deg', type=float, nargs='+', default=[2., 5.])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error('Gürültü tohumları tekil olmalı.')
    if any(not math.isfinite(yaw) or not 0 < abs(yaw) <= 10 for yaw in args.fault_yaw_deg):
        parser.error('Hata açılarının mutlak değeri 0–10° arasında olmalı; sıfır hariç.')
    yaws = [0.] + list(dict.fromkeys(args.fault_yaw_deg))
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root/'results'/f'exp_28_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    total = 2 * len(args.seeds) * len(yaws) * len(POLICIES)
    print(f'EXP-28: {total} kısa deneme; aynı NIS eşiği ve hareket seçici korunuyor.')
    rows, checks = [], {}
    trial_index = 0
    for normal in ('straight', 'arc'):
        for seed in args.seeds:
            for yaw in yaws:
                paired_histories, triggers = {}, {}
                condition = 'healthy' if yaw == 0. else f'front_yaw_{yaw:g}deg'.replace('-','minus').replace('.','p')
                for policy in POLICIES:
                    agent, events, commands, records, completed, stop_reason = run_trial(policy, normal, yaw, seed)
                    folder = output/f'normal_{normal}'/f'seed_{seed}'/condition/policy
                    folder.mkdir(parents=True)
                    write_jsonl(folder/'measurements.jsonl', events)
                    truth_path = folder/'ground_truth.csv'
                    write_csv(truth_path, records['truth'])
                    write_csv(folder/'commands.csv', commands)
                    write_csv(folder/'agent_history.csv', agent.history)
                    write_jsonl(folder/'mounting_hypotheses.jsonl', agent.hypotheses)
                    write_json(folder/'decision.json', agent.decision)
                    write_json(folder/'simulation_metadata.json', records['metadata'])
                    _, evaluated, metrics = evaluate({'line_slam': agent.path}, truth_path)
                    write_csv(folder/'line_slam.csv', evaluated['line_slam'])
                    trigger = agent.hold.trigger_time_s
                    premature = yaw != 0. and trigger is not None and trigger < 2. - 1e-9
                    delay = trigger - 2. if yaw != 0. and trigger is not None and not premature else None
                    selected = agent.selected_motion
                    row = {
                        'normal_motion': normal, 'policy': policy, 'seed': seed,
                        'fault_yaw_deg': yaw,
                        'completed': completed, 'stop_reason': stop_reason,
                        'alarm': agent.hold.active, 'trigger_time_s': trigger,
                        'premature_alarm': premature, 'detection_delay_s': delay,
                        'selected_motion': selected,
                        'full_position_rmse_m': metrics['line_slam']['position_rmse_m'],
                        'full_heading_rmse_rad': metrics['line_slam']['heading_rmse_rad'],
                        'changed_command_intervals': sum(c['applied'] and c['changed_from_normal'] for c in commands),
                        **evidence_metrics(agent.hypotheses),
                    }
                    rows.append(row)
                    write_json(folder/'summary.json', row)
                    triggers[policy] = trigger
                    paired_histories[policy] = events
                    trial_index += 1
                    print(f'[{trial_index}/{total}] normal={normal} seed={seed} {condition} {policy}: alarm={agent.hold.active}, hareket={selected}, tamamlandı={completed}', flush=True)
                # Politikalar ancak ilk alarmdan sonra ayrılabilir.
                key = f'{normal}_seed{seed}_{condition}'
                same_trigger = len(set(triggers.values())) == 1
                checks[f'{key}_same_first_alarm_time'] = same_trigger
                if not same_trigger:
                    raise RuntimeError(f'Eşleştirme kontrolü: ilk alarm zamanları farklı: {key}')
                end = next(iter(triggers.values()))
                end = 4. if end is None else end
                prefixes = [[e for e in events if e['time_s'] <= end + 1e-9] for events in paired_histories.values()]
                same_prefix = all(prefix == prefixes[0] for prefix in prefixes[1:])
                checks[f'{key}_same_measurements_until_alarm'] = same_prefix
                if not same_prefix:
                    raise RuntimeError(f'Eşleştirme kontrolü: alarm öncesi kayıtlar farklı: {key}')
    summaries = group_results(rows)
    write_csv(output/'trials.csv', rows)
    write_csv(output/'group_summary.csv', summaries)
    write_json(output/'summary.json', {
        'experiment_id':'exp_28', 'trial_count':len(rows),
        'seeds':args.seeds, 'fault_yaw_deg':yaws,
        'nis_threshold':9.21034, 'checks':checks,
        'all_pairing_checks_passed':all(checks.values()),
        'groups':summaries,
        'limitations':[
            'İki normal rota aynı tek ortamda kullanılır; farklı ortam genellemesi değildir.',
            'Üç varsayılan tohum keşif amaçlıdır; nüfus başarım oranı veya anlamlılık testi değildir.',
            'Sağlıklı alarm ve hatalı durumda alarm yokluğu sayıları ayrı raporlanır.',
            'Alarm hata kaynağının doğru teşhis edildiği anlamına gelmez.',
            'Ön hipotez farkı kalibre edilmiş doğruluk değil, koşullu uyum farkıdır.',
            'Skor ortalaması yalnız destekli anlardan hesaplanır; destek/total birlikte okunmalıdır.',
            'Tamamlanmayan denemelerin konum RMSE değeri grup ortalamasına katılmaz; sayıları korunur.',
            'NIS eşiği ve hareket seçici mevcut sonuçlara göre değiştirilmemiştir.',
            'GPS kaydedilir, bu karşılaştırmada füzyona veya teşhise alınmaz.',
        ],
    })
    print('\nGRUP ÖZETİ:')
    for group in summaries:
        rmse=group['mean_full_position_rmse_m']
        text=f'{rmse:.6f}' if rmse is not None else 'yok'
        print(
            f"{group['normal_motion']}/{group['policy']}/{group['fault_yaw_deg']:g}deg: "
            f"alarm={group['alarm_trials']}/{group['trials']} | "
            f"konum RMSE={text} m | destek={group['supported_frames']}/{group['hypothesis_frames']} | "
            f"komut değiştiren deneme={group['trials_with_changed_commands']}"
        )
    figure, axes = plt.subplots(1,2,figsize=(12,4.8))
    for axis, normal in zip(axes,('straight','arc')):
        for policy in POLICIES:
            groups=sorted((g for g in summaries if g['normal_motion']==normal),key=lambda g:g['fault_yaw_deg'])
            selected=[g for g in groups if g['policy']==policy]
            axis.plot([g['fault_yaw_deg'] for g in selected],[g['mean_full_position_rmse_m'] for g in selected],marker='o',label=policy)
        axis.set(title=f'Normal rota: {normal}',xlabel='Ön montaj hatası (derece)',ylabel='Ortalama konum RMSE (m)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output/'comparison.png',dpi=160)
    plt.close(figure)
    print('Alarm öncesi eşleştirilmiş kayıt kontrolleri: PASS')
    print(f'Sonuç klasörü: {output}')


if __name__=='__main__':
    main()
