"""EXP-29: Alarm geç gelse de iki saniyelik teşhis penceresini tamamlar.

Normal gözlem ufku 4 saniyedir. Bu ufuk içinde alarm oluşursa simülasyon,
alarm+2 saniyeye kadar uzar. Konum hatası ortak [0,4] ufkunda; tanısal
kanıt ve ek konum hatası yalnız (alarm,alarm+2] aralığında değerlendirilir.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path

from storage.files import write_json, write_jsonl
from experiments.exp_28_policy_benchmark import make_agent, POLICIES
from evaluation.trajectory import evaluate
from storage.files import write_csv
from simulator.streaming_world import StreamingWorld, WorldConfig

BASE_HORIZON = 4.0


def drive_trial(policy, normal_motion, yaw, seed, *, world_factory=StreamingWorld, agent_factory=make_agent):
    world = world_factory(WorldConfig(front_yaw_error_deg=yaw, gyro_seed=seed, gps_seed=seed+1))
    agent = agent_factory(policy, normal_motion)
    events = world.initial_measurements()
    agent.process(events)
    deadline = BASE_HORIZON
    commands, blocked, stop_reason = [], False, None
    while agent.estimator.time_s < deadline - 1e-9:
        rates = agent.command()
        row = {
            'interval_start_s': agent.estimator.time_s,
            'left_rate_rad_s': rates[0], 'right_rate_rad_s': rates[1],
            'mode_before_step': agent.mode,
            'changed_from_normal': rates != agent.normal_command,
            'applied': False,
        }
        commands.append(row)
        try:
            batch = world.step(*rates)
        except RuntimeError as error:
            blocked, stop_reason = True, str(error)
            break
        row['applied'] = True
        agent.process(batch)
        events.extend(batch)
        if agent.hold.active:
            # Alarm yalnız ölçümlerden gelir. Gerçek hata başlangıcına bakılmaz.
            deadline = max(BASE_HORIZON, agent.window_end_s)
        if deadline > BASE_HORIZON + 2. + 1e-8:
            raise RuntimeError('Beklenmeyen teşhis son zamanı.')
    time_s = agent.estimator.time_s
    trigger = agent.hold.trigger_time_s
    checks = {
        'common_horizon_completed': time_s >= BASE_HORIZON - 1e-9,
        'diagnostic_window_completed': trigger is not None and time_s >= agent.window_end_s - 1e-9,
        'required_horizon_completed': not blocked and time_s >= deadline - 1e-9,
    }
    return agent, events, commands, world.evaluation_records(), checks, stop_reason


def window_evidence(hypotheses, trigger_time_s, window_end_s):
    if trigger_time_s is None:
        return {'window_scan_instants':0, 'supported_frames':0, 'front_margin_sum':0., 'mean_front_margin':None}
    frames = [
        row for row in hypotheses
        if row['time_s'] > trigger_time_s + 1e-9 and row['time_s'] <= window_end_s + 1e-9
    ]
    supported = [row for row in frames if row['status']=='evaluated']
    margins = []
    for row in supported:
        models = {model['hypothesis']:model for model in row['models']}
        margins.append(models['front_yaw']['nll_improvement_over_nominal'] - max(
            models[name]['nll_improvement_over_nominal'] for name in ('rear_yaw','heading_offset')
        ))
    return {
        'window_scan_instants':len(frames), 'supported_frames':len(supported),
        'front_margin_sum':sum(margins),
        'mean_front_margin':sum(margins)/len(margins) if margins else None,
    }


def position_rmse(rows):
    return math.sqrt(sum(row['position_error_m']**2 for row in rows)/len(rows)) if rows else None


def summarize_trials(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['normal_motion'],row['policy'],row['fault_yaw_deg'])].append(row)
    summary=[]
    for (normal,policy,yaw), trials in sorted(groups.items()):
        common=[r for r in trials if r['common_horizon_completed']]
        windows=[r for r in trials if r['diagnostic_window_completed']]
        support=sum(r['supported_frames'] for r in windows)
        summary.append({
            'normal_motion':normal, 'policy':policy, 'fault_yaw_deg':yaw,
            'trials':len(trials), 'alarm_trials':sum(r['alarm'] for r in trials),
            'common_horizon_completed':len(common),
            'complete_diagnostic_windows':len(windows),
            'mean_common_position_rmse_m':sum(r['common_position_rmse_m'] for r in common)/len(common) if common else None,
            'mean_diagnostic_position_rmse_m':sum(r['diagnostic_position_rmse_m'] for r in windows)/len(windows) if windows else None,
            'supported_frames':support,
            'window_scan_instants':sum(r['window_scan_instants'] for r in windows),
            'pooled_front_margin':sum(r['front_margin_sum'] for r in windows)/support if support else None,
            'trials_with_changed_diagnostic_commands':sum(r['changed_diagnostic_command_intervals']>0 for r in trials),
        })
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    parser.add_argument('--fault-yaw-deg',type=float,nargs='+',default=[2.,5.])
    parser.add_argument('--normal-motion',choices=['straight','arc'],nargs='+',default=['straight','arc'])
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.normal_motion))!=len(args.normal_motion):
        parser.error('Tohumlar ve normal rotalar tekil olmalı.')
    if any(not math.isfinite(yaw) or not 0<abs(yaw)<=10 for yaw in args.fault_yaw_deg):
        parser.error('Hata açılarının mutlak değeri 0–10° arasında olmalı; sıfır hariç.')
    yaws=[0.]+list(dict.fromkeys(args.fault_yaw_deg))
    root=Path(__file__).resolve().parents[1]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=root/'results'/f'exp_29_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    total=len(args.normal_motion)*len(args.seeds)*len(yaws)*len(POLICIES)
    print(f'EXP-29: {total} deneme; her alarmdan sonra tam 2 saniye gözlem.')
    rows, pairing_checks=[],{}
    for normal in args.normal_motion:
        for seed in args.seeds:
            for yaw in yaws:
                histories,triggers={},{}
                condition='healthy' if yaw==0. else f'front_yaw_{yaw:g}deg'.replace('-','minus').replace('.','p')
                for policy in POLICIES:
                    agent,events,commands,records,checks,reason=drive_trial(policy,normal,yaw,seed)
                    folder=output/f'normal_{normal}'/f'seed_{seed}'/condition/policy
                    folder.mkdir(parents=True)
                    write_jsonl(folder/'measurements.jsonl',events)
                    truth_path=folder/'ground_truth.csv'
                    write_csv(truth_path,records['truth'])
                    write_csv(folder/'commands.csv',commands)
                    write_csv(folder/'agent_history.csv',agent.history)
                    write_jsonl(folder/'lidar_innovations.jsonl',agent.innovations)
                    write_jsonl(folder/'mounting_hypotheses.jsonl',agent.hypotheses)
                    write_json(folder/'decision.json',agent.decision)
                    write_json(folder/'simulation_metadata.json',records['metadata'])
                    _,evaluated,_=evaluate({'line_slam':agent.path},truth_path)
                    estimated=evaluated['line_slam']
                    write_csv(folder/'line_slam.csv',estimated)
                    trigger=agent.hold.trigger_time_s
                    end=agent.window_end_s
                    diagnostic=[r for r in estimated if trigger is not None and trigger+1e-9<r['time_s']<=end+1e-9]
                    evidence=window_evidence(agent.hypotheses,trigger,end)
                    row={
                        'normal_motion':normal,'policy':policy,'seed':seed,'fault_yaw_deg':yaw,
                        'alarm':agent.hold.active,'trigger_time_s':trigger,
                        'selected_motion':agent.selected_motion,
                        'simulation_end_s':agent.estimator.time_s,
                        'diagnostic_window_end_s':end,
                        'extra_simulation_time_s':max(0.,agent.estimator.time_s-BASE_HORIZON),
                        **checks,'stop_reason':reason,
                        'common_position_rmse_m':position_rmse([r for r in estimated if r['time_s']<=BASE_HORIZON+1e-9]) if checks['common_horizon_completed'] else None,
                        'diagnostic_position_rmse_m':position_rmse(diagnostic) if checks['diagnostic_window_completed'] else None,
                        'changed_diagnostic_command_intervals':sum(
                            c['applied'] and c['changed_from_normal'] and trigger is not None
                            and c['interval_start_s']>=trigger-1e-9
                            and c['interval_start_s']<end-1e-9 for c in commands
                        ),
                        **evidence,
                    }
                    if checks['diagnostic_window_completed'] and evidence['window_scan_instants']!=4:
                        raise RuntimeError('İki saniyelik pencerede dört tarama anı bulunmalı.')
                    rows.append(row)
                    write_json(folder/'summary.json',{**row,'trigger_evidence':agent.hold.trigger_evidence})
                    histories[policy],triggers[policy]=events,trigger
                    print(
                        f"[{len(rows)}/{total}] {normal}/{seed}/{condition}/{policy}: "
                        f"alarm={trigger}, bitiş={agent.estimator.time_s:.1f}, "
                        f"tanı desteği={evidence['supported_frames']}/{evidence['window_scan_instants']}, "
                        f"değişen tanı komutu={row['changed_diagnostic_command_intervals']}"
                    ,flush=True)
                key=f'{normal}_{seed}_{condition}'
                same_trigger=len(set(triggers.values()))==1
                pairing_checks[f'{key}_same_first_alarm']=same_trigger
                if not same_trigger:
                    raise RuntimeError('İlk alarm zamanı politikalar arasında farklı.')
                cutoff=next(iter(triggers.values()))
                cutoff=BASE_HORIZON if cutoff is None else cutoff
                prefixes=[[e for e in events if e['time_s']<=cutoff+1e-9] for events in histories.values()]
                same_prefix=all(prefix==prefixes[0] for prefix in prefixes[1:])
                pairing_checks[f'{key}_same_prefix']=same_prefix
                if not same_prefix:
                    raise RuntimeError('Alarm öncesindeki ölçümler eşleşmiyor.')
    groups=summarize_trials(rows)
    write_csv(output/'trials.csv',rows)
    write_csv(output/'group_summary.csv',groups)
    write_json(output/'summary.json',{
        'experiment_id':'exp_29','trial_count':len(rows),'groups':groups,
        'pairing_checks':pairing_checks,
        'all_required_horizons_completed':all(r['required_horizon_completed'] for r in rows),
        'protocol':{
            'normal_observation_horizon_s':BASE_HORIZON,'diagnosis_duration_s':2.,
            'common_localization_evaluation':'[0,4]',
            'diagnostic_evaluation':'(first_alarm,first_alarm+2]',
        },
        'limitations':[
            'Alarm yoksa tanısal pencere ve tanısal skor yoktur; sıfır puan yazılmaz.',
            'Tanısal pencereler farklı tohumlarda farklı zamanlarda başlayabilir; politikalar eşleştirilmiştir.',
            'Alarm öncesi ve alarmdan sonraki belirtiler aynı ortalamaya karıştırılmaz.',
            'Yerelleştirme ortak 4 saniyelik ufukta ayrıca raporlanır; değişken kayıt süreleri doğrudan birleştirilmez.',
            'Alarm kaynağının kesin teşhisi veya kalibre edilmiş hata olasılığı üretilmez.',
            'NIS eşiği, hipotez bankası ve hareket seçici değiştirilmemiştir.',
        ],
    })
    print('Ortak geçmiş ve ilk alarm eşleştirmeleri: PASS')
    print(f'Sonuç klasörü: {output}')


if __name__=='__main__':
    main()
