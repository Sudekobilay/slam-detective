"""EXP-30: Sağlıklı kalibrasyonla ek tarama izleyicisinin sabit-kayıt denetimi.

Alarmı ve geçici harita korumasını sınar. İki yöntem aynı dört saniyelik ham
ölçümleri alır; bu deney hareket seçimi başarımı karşılaştırması değildir.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

from estimation.online_diagnostic_agent import OnlineDiagnosticAgent
from estimation.scan_alignment_guard import ScanAlignmentGuard,ScanGuardConfig
from experiments.exp_24_motion_comparison import write_json,write_jsonl
from experiments.line_slam_common import evaluate,write_csv
from simulator.streaming_world import StreamingWorld,WorldConfig


CALIBRATION_SEEDS=tuple(range(11,21))
REGRESSION_SEEDS=(101,202,303)
HELD_OUT_SEEDS=(401,402,403,404,405)
NORMAL_COMMANDS={'straight':(4.,4.),'arc':(3.1,4.9)}


def fixed_recording(normal_motion,seed,yaw):
    world=StreamingWorld(WorldConfig(front_yaw_error_deg=yaw,gyro_seed=seed,gps_seed=seed+1))
    events=world.initial_measurements()
    for _ in range(40):
        events.extend(world.step(*NORMAL_COMMANDS[normal_motion]))
    return events,world.evaluation_records()


def replay_monitor(events,normal_motion,config=None):
    guard=ScanAlignmentGuard(config) if config is not None else None
    agent=OnlineDiagnosticAgent(strategy='passive',scan_guard=guard)
    agent.normal_command=NORMAL_COMMANDS[normal_motion]
    for _,batch in groupby(events,key=lambda event:event['time_s']):
        agent.process(list(batch))
    return agent


def calibrate_healthy(seeds=CALIBRATION_SEEDS):
    # Gevşek bir ölçüm toplama yapılandırması; bu aşamada tarama izleyicisi
    # uyarı veya kilit üretmez. Gerçek hata kaydı kullanılmaz.
    collection=ScanGuardConfig(relative_yaw_threshold_deg=19.,max_alignment_rms_m=100.)
    samples=[]
    healthy_nis_alarms=0
    for normal in NORMAL_COMMANDS:
        for seed in seeds:
            events,_=fixed_recording(normal,seed,0.)
            agent=replay_monitor(events,normal,collection)
            healthy_nis_alarms+=int(agent.hold.active)
            for frame in agent.scan_guard.history:
                alignments=frame['alignments']
                if not all(alignments.get(s) is not None and not alignments[s]['at_search_edge'] for s in ('front','rear')):
                    continue
                samples.append({
                    'normal_motion':normal,'seed':seed,'time_s':frame['time_s'],
                    'relative_yaw_deg':alignments['front']['offset_deg']-alignments['rear']['offset_deg'],
                    'max_alignment_rms_m':max(alignments[s]['alignment_rms_m'] for s in ('front','rear')),
                })
    if not samples:
        raise ValueError('Sağlıklı kalibrasyonda destekli örnek bulunamadı.')
    maximum_gap=max(abs(row['relative_yaw_deg']) for row in samples)
    maximum_rms=max(row['max_alignment_rms_m'] for row in samples)
    # Marjlar önceden seçilmiş mühendislik paylarıdır; %99 gibi bir güven
    # düzeyi iddia edilmez. Değerler hata kayıtlarından ayarlanmaz.
    config=ScanGuardConfig(
        relative_yaw_threshold_deg=max(.5,maximum_gap+.5),
        max_alignment_rms_m=maximum_rms+.01,
    )
    return config,{
        'calibration_seeds':list(seeds),'healthy_sequences':2*len(seeds),
        'supported_scan_instants':len(samples),'maximum_healthy_relative_yaw_deg':maximum_gap,
        'maximum_healthy_alignment_rms_m':maximum_rms,
        'angle_margin_deg':.5,'rms_margin_m':.01,
        'healthy_nis_alarms_during_collection':healthy_nis_alarms,
        'config':asdict(config),
    },samples


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['split'],row['normal_motion'],row['monitor'],row['fault_yaw_deg'])].append(row)
    summaries=[]
    for (split,normal,monitor,yaw),trials in sorted(groups.items()):
        valid_delays=[r['detection_delay_s'] for r in trials if r['detection_delay_s'] is not None]
        summaries.append({
            'split':split,'normal_motion':normal,'monitor':monitor,'fault_yaw_deg':yaw,
            'trials':len(trials),'alarm_trials':sum(r['alarm'] for r in trials),
            'healthy_alarm_trials':sum(r['alarm'] for r in trials) if yaw==0 else 0,
            'fault_trials_without_alarm':sum(not r['alarm'] for r in trials) if yaw!=0 else 0,
            'post_fault_alarms':len(valid_delays),
            'premature_alarms':sum(r['premature_alarm'] for r in trials),
            'mean_detection_delay_s':sum(valid_delays)/len(valid_delays) if valid_delays else None,
            'mean_common_position_rmse_m':sum(r['position_rmse_m'] for r in trials)/len(trials),
            'trials_with_quarantine':sum(r['quarantined_scans']>0 for r in trials),
        })
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held-out-seeds',type=int,nargs='+',default=list(HELD_OUT_SEEDS))
    args=parser.parse_args()
    test_seeds=tuple(args.held_out_seeds)
    if not test_seeds or len(test_seeds)!=len(set(test_seeds)):
        parser.error('Ayrı test tohumları tekil ve boş olmayan bir liste olmalı.')
    if set(test_seeds)&(set(CALIBRATION_SEEDS)|set(REGRESSION_SEEDS)):
        parser.error('Ayrı test, kalibrasyon ve önceki regresyon tohumlarından farklı olmalı.')
    root=Path(__file__).resolve().parents[1]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=root/'results'/f'exp_30_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    print('EXP-30: yalnız sağlıklı kalibrasyon; aynı kayıtta özgün NIS ve ek ham tarama izleyicisi.')
    config,calibration,samples=calibrate_healthy()
    write_json(output/'calibration.json',calibration)
    write_csv(output/'calibration_samples.csv',samples)
    print(f'Kalibrasyon: göreli açı eşiği={config.relative_yaw_threshold_deg:.3f} derece; uyum RMS sınırı={config.max_alignment_rms_m:.6f} m')
    rows=[]
    for split,seeds in (('regression',REGRESSION_SEEDS),('held_out',test_seeds)):
        for normal in NORMAL_COMMANDS:
            for seed in seeds:
                for yaw in (0.,2.,5.):
                    case='healthy' if yaw==0 else f'front_yaw_{yaw:g}deg'
                    folder=output/split/f'normal_{normal}'/f'seed_{seed}'/case
                    folder.mkdir(parents=True)
                    events,records=fixed_recording(normal,seed,yaw)
                    source=folder/'measurements.jsonl'
                    truth=folder/'ground_truth.csv'
                    write_jsonl(source,events)
                    write_csv(truth,records['truth'])
                    write_json(folder/'simulation_metadata.json',records['metadata'])
                    digest=hashlib.sha256(source.read_bytes()).hexdigest()
                    # Hata etiketi, seed ve gerçek konum replay_monitor'a verilmez.
                    for monitor,cfg in (('matched_nis',None),('scan_consensus',config)):
                        agent=replay_monitor(events,normal,cfg)
                        _,evaluated,metrics=evaluate({'line_slam':agent.path},truth)
                        dest=folder/monitor
                        dest.mkdir()
                        write_csv(dest/'line_slam.csv',evaluated['line_slam'])
                        write_csv(dest/'agent_history.csv',agent.history)
                        write_jsonl(dest/'lidar_innovations.jsonl',agent.innovations)
                        write_jsonl(dest/'scan_alignment.jsonl',agent.scan_guard.history if agent.scan_guard else [])
                        trigger=agent.hold.trigger_time_s
                        premature=yaw!=0 and trigger is not None and trigger<2.-1e-9
                        row={
                            'split':split,'normal_motion':normal,'seed':seed,'fault_yaw_deg':yaw,'monitor':monitor,
                            'alarm':agent.hold.active,'trigger_time_s':trigger,
                            'trigger_source':getattr(agent.hold,'trigger_source','matched_nis' if agent.hold.active else None),
                            'premature_alarm':premature,
                            'detection_delay_s':trigger-2. if yaw!=0 and trigger is not None and not premature else None,
                            'position_rmse_m':metrics['line_slam']['position_rmse_m'],
                            'quarantined_scans':agent.counts['quarantined_scans'],
                            'held_scans':agent.counts['held_scans'],
                            'landmark_count':agent.estimator.landmark_count,
                        }
                        rows.append(row)
                        write_json(dest/'summary.json',{**row,'trigger_evidence':agent.hold.trigger_evidence,'input_sha256':digest})
                    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
                        raise RuntimeError('Ortak ölçüm kaydı değişti.')
            print(f'{split}/{normal}: {len(seeds)} tohum tamamlandı.',flush=True)
    summary=summarize(rows)
    write_csv(output/'trials.csv',rows)
    write_csv(output/'group_summary.csv',summary)
    write_json(output/'summary.json',{
        'experiment_id':'exp_30','calibration':calibration,'groups':summary,
        'regression_seeds':list(REGRESSION_SEEDS),'held_out_seeds':list(test_seeds),
        'nis_threshold_unchanged':9.21034,
        'limitations':[
            'Bu sabit-kayıt alarm/harita-koruma denetimidir; hareket seçimi yararı ölçülmez.',
            'İki izleyici aynı gerçek ölçümleri alır; geçici harita koruması tahminlerini değiştirebilir.',
            'Göreli açı tutarsızlığı hata kaynağını kesin belirlemez.',
            'Mühendislik marjları ve kırpılmış nokta uyumu kullanılır; istatistiksel güven düzeyi iddia edilmez.',
            'Kalibrasyon ve ayrı test tek ortam ve ideal LiDAR gürültü modeliyle sınırlıdır.',
            'Kalibrasyon etiketleri yalnız sağlıklı kayıt seçiminde kullanılır; çalışma zamanı hata etiketi almaz.',
            'Geçici koruma yeterli uyum desteği kaybolursa sürebilir; otomatik kurtarma tasarımı tamamlanmamıştır.',
            'Mevcut deneylerin varsayılan davranışı matched_nis olarak korunur.',
        ],
    })
    for group in summary:
        print(f"{group['split']}/{group['normal_motion']}/{group['fault_yaw_deg']:g}deg/{group['monitor']}: alarm={group['alarm_trials']}/{group['trials']}, erken={group['premature_alarms']}, gecikme={group['mean_detection_delay_s']}")
    print(f'Sonuç klasörü: {output}')


if __name__=='__main__':
    main()
