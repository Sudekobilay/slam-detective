"""EXP-42: Dondurulmuş kalibrasyonla iki ek ortamda dört hareket politikası.

Mevcut çekirdekler değiştirilmez. Gerçek ortam ve hata etiketi sadece ölçüm
üretimi/çevrimdışı değerlendirmededir. Yeni ortamlara eşik uyarlaması yoktur.
"""

from application.calibration import calibrate_readout

from evaluation.diagnosis import check_pairing, diagnosis_readout, outcome
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics

from estimation.diagnostic_motion import CANDIDATES
from estimation.raw_scan_hypotheses import diagnose_raw
from storage.files import write_json, write_jsonl
from evaluation.movement import movement_metrics
from configuration.layouts import examples
from simulator.layout_world import LayoutWorld
from application.layout_service import calibrate_layout, create_agent, run_layout
from evaluation.trajectory import evaluate
from storage.files import write_csv
from simulator.collision import try_step
from simulator.robot import DifferentialDriveRobot, Pose
from simulator.validation_environments import validation_environment


ENVIRONMENTS = ('corridor', 'offset_obstacles')
POLICIES = ('passive', 'fixed_straight', 'fixed_arc', 'adaptive')
CALIBRATION_SEEDS = tuple(range(11,16))
DEFAULT_SEEDS = tuple(range(1201,1211))
ANGLES = (0.,2.,5.)
PROTOCOL = {
    'experiment_id':'exp_42', 'algorithm':'ekf', 'layout':'quad', 'gps_enabled':False,
    'environments':list(ENVIRONMENTS), 'normal_motions':['straight','arc'],
    'fault_angles_deg':list(ANGLES), 'fault_sensor':'front', 'fault_start_s':2.,
    'policies':list(POLICIES), 'base_horizon_s':4., 'diagnostic_window_s':2.,
    'calibration_environment':'original_demo_only', 'calibration_seeds':list(CALIBRATION_SEEDS),
    'primary_endpoint':'correct_front_yaw_confirmation_within_post_alarm_2s_window_among_all_fault_trials',
    'confirmation_rule':'same_model_and_offset_sign_for_two_consecutive_supported_post_alarm_scans',
    'primary_denominator':'all_fault_trials; no_alarm, premature_alarm, incomplete_window and uncertainty count as no correct confirmation',
    'secondary_metrics':['healthy_false_alarm','alarm_delay','uncertainty','wrong_source','common_position_rmse_m',
                         'full_support','mean_true_source_margin_m2','movement_distance','changed_command_duration'],
    'no_posthoc_threshold_tuning':True,
    'limitations':['Two extra geometries, one initial pose, front mounting fault only, positive 2/5 degrees, ideal LiDAR.',
                   'No universal generalization or statistical significance claim.',
                   'This is the EXP-40 four-LiDAR alarm/model pipeline, not the earlier EXP-33 two-LiDAR protocol.'],
}


class ValidationWorld(LayoutWorld):
    def __init__(self, environment, data, seed, yaw):
        super().__init__(data,seed,'healthy' if yaw==0 else 'lidar','front',yaw, environment=validation_environment(environment))
        self.environment_name = environment

    def evaluation_records(self):
        result = super().evaluation_records()
        result['metadata']['validation_environment'] = self.environment_name
        return result


def preflight_geometry():
    """Hatasız fizik: normal rota ve tüm olası iki saniyelik aday dalları."""
    checks = {}
    for environment in ENVIRONMENTS:
        world = validation_environment(environment)
        safe = True
        for normal in CANDIDATES.values():
            robot = DifferentialDriveRobot(initial_pose=Pose(1.,3.,0.))
            poses = [robot.pose]
            for _ in range(40):
                safe = try_step(robot,world,*normal,.1,radius=.35) and safe
                poses.append(robot.pose)
            for pose in poses:
                for rates in CANDIDATES.values():
                    branch = DifferentialDriveRobot(initial_pose=pose)
                    for _ in range(20):
                        safe = try_step(branch,world,*rates,.1,radius=.35) and safe
        checks[environment] = safe
    return checks




def run_trial(environment,data,config,policy,normal,yaw,seed):
    world = ValidationWorld(environment,data,seed,yaw)
    agent = create_agent(data,config,'ekf',policy=policy,normal=normal,seed=seed+20000,use_gps=False)
    from application.runner import run_episode
    return run_episode(world, agent).legacy()


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row['environment'],row['normal_motion'],row['fault_yaw_deg'],row['policy']].append(row)
    result=[]
    for (environment,normal,yaw,policy),trials in sorted(groups.items()):
        margins=[r['mean_true_source_margin_m2'] for r in trials if r['mean_true_source_margin_m2'] is not None]
        result.append({
            'environment':environment,'normal_motion':normal,'fault_yaw_deg':yaw,'policy':policy,'trials':len(trials),
            **{name:sum(r['outcome']==name for r in trials) for name in
               ('correct_source','wrong_source','uncertain','no_alarm','premature_alarm','incomplete_window','healthy_no_alarm','false_alarm')},
            'alarm_trials':sum(r['trigger_time_s'] is not None for r in trials),
            'correct_source_rate':sum(r['outcome']=='correct_source' for r in trials)/len(trials) if yaw else None,
            'mean_margin_m2':statistics.mean(margins) if margins else None,
            'margin_supported_trials':len(margins),
            'changed_command_trials':sum((r['changed_command_duration_s'] or 0)>1e-9 for r in trials),
            'completed_trials':sum(r['required_horizon_completed'] for r in trials),
        })
    return result


def paired_summary(rows):
    groups=defaultdict(dict)
    for r in rows:
        key=(r['environment'],r['normal_motion'],r['fault_yaw_deg'],r['seed'])
        groups[key][r['policy']]=r
    combined=defaultdict(list)
    for (environment,normal,yaw,seed),policies in groups.items():
        if not yaw:
            continue
        adaptive=policies['adaptive']
        for comparator in ('passive','fixed_straight','fixed_arc'):
            other=policies[comparator]
            combined[environment,normal,yaw,comparator].append((adaptive,other))
    result=[]
    for (environment,normal,yaw,comparator),pairs in sorted(combined.items()):
        margins=[a['mean_true_source_margin_m2']-b['mean_true_source_margin_m2'] for a,b in pairs
                 if a['mean_true_source_margin_m2'] is not None and b['mean_true_source_margin_m2'] is not None]
        result.append({'environment':environment,'normal_motion':normal,'fault_yaw_deg':yaw,'comparator':comparator,
                       'paired_trials':len(pairs),'adaptive_only_correct':sum(a['outcome']=='correct_source' and b['outcome']!='correct_source' for a,b in pairs),
                       'comparator_only_correct':sum(a['outcome']!='correct_source' and b['outcome']=='correct_source' for a,b in pairs),
                       'both_correct':sum(a['outcome']==b['outcome']=='correct_source' for a,b in pairs),
                       'neither_correct':sum(a['outcome']!='correct_source' and b['outcome']!='correct_source' for a,b in pairs),
                       'margin_pairs':len(margins),'mean_margin_difference_m2':statistics.mean(margins) if margins else None})
    return result


from storage.provenance import protected_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(DEFAULT_SEEDS))
    parser.add_argument('--save-recordings',action='store_true',help='Her denemenin ham ölçümlerini de sakla.')
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds or set(args.seeds)&set(CALIBRATION_SEEDS):
        parser.error('Tohumlar tekil ve kalibrasyondan ayrı olmalı.')
    root=Path(__file__).resolve().parents[1]
    before=protected_hashes(root)
    output=root/'results'/('exp_42_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    data={**examples()['quad'],'gps_enabled':False}
    checks=preflight_geometry()
    write_json(output/'protocol.json',{**PROTOCOL,'evaluation_seeds':args.seeds,'preflight':checks,
               'trial_count':len(ENVIRONMENTS)*2*len(ANGLES)*len(POLICIES)*len(args.seeds),
               'environment_geometry':{name:asdict(validation_environment(name)) for name in ENVIRONMENTS}})
    if not all(checks.values()):
        raise RuntimeError('Hatasız aday hareket geometrisi güvenli değil.')
    print('EXP-42: iki ek ortam; dört LiDAR + EKF, GPS kapalı; eşikler yalnız eski ortamdan.',flush=True)
    config,thresholds,max_rms,calibration=calibrate_readout(data)
    write_json(output/'calibration.json',calibration)
    write_json(output/'used_config.json',data)
    frozen=hashlib.sha256((output/'calibration.json').read_bytes()).hexdigest()
    rows,pairing=[],{}
    for environment in ENVIRONMENTS:
        for normal in CANDIDATES:
            for seed in args.seeds:
                for yaw in ANGLES:
                    trials={policy:run_trial(environment,data,config,policy,normal,yaw,seed) for policy in POLICIES}
                    pair_checks=check_pairing(trials)
                    pairing[f'{environment}/{normal}/{seed}/{yaw:g}']=pair_checks
                    if not all(pair_checks.values()):
                        raise RuntimeError('Politikaların alarm öncesi geçmişi eşleşmiyor.')
                    for policy,(agent,events,commands,records,reason) in trials.items():
                        folder=output/environment/normal/f'seed_{seed}'/f'yaw_{yaw:g}'/policy
                        folder.mkdir(parents=True)
                        trigger=agent.trigger_time_s
                        complete=reason is None and agent.time_s>=max(4.,agent.window_end_s or 4.)-1e-9
                        diagnostic=trigger is not None and agent.time_s>=agent.window_end_s-1e-9
                        evidence=diagnosis_readout(agent.raw_history,trigger,agent.window_end_s,thresholds,max_rms)
                        truth=folder/'ground_truth.csv'
                        write_csv(truth,records['truth'])
                        _,evaluated,metrics=evaluate({'line_slam':agent.path},truth)
                        common=[r['position_error_m'] for r in evaluated['line_slam'] if r['time_s']<=4.+1e-9]
                        row={'environment':environment,'normal_motion':normal,'seed':seed,'fault_yaw_deg':yaw,'policy':policy,
                             'trigger_time_s':trigger,'selected_motion':agent.selected_motion,
                             'required_horizon_completed':complete,'diagnostic_window_completed':diagnostic,
                             'stop_reason':reason,'outcome':outcome(yaw,trigger,diagnostic and complete,evidence['confirmed_model']),
                             'common_position_rmse_m':math.sqrt(sum(e*e for e in common)/len(common)) if agent.time_s>=4.-1e-9 else None,
                             **evidence,**movement_metrics(records['truth'],commands,trigger,agent.window_end_s)}
                        rows.append(row)
                        write_json(folder/'summary.json',row)
                        write_json(folder/'decision.json',agent.decision)
                        write_jsonl(folder/'raw_hypotheses.jsonl',agent.raw_history)
                        write_csv(folder/'commands.csv',commands)
                        if args.save_recordings:
                            write_jsonl(folder/'measurements.jsonl',events)
                            write_json(folder/'simulation_metadata.json',records['metadata'])
                write_csv(output/'trials.csv',rows)
                print(f'{environment}/{normal}/seed_{seed}: {len(rows)} deneme tamamlandı.',flush=True)
    group_rows=summarize(rows)
    paired=paired_summary(rows)
    write_csv(output/'group_summary.csv',group_rows)
    write_csv(output/'paired_summary.csv',paired)
    unchanged=protected_hashes(root)==before
    frozen_unchanged=hashlib.sha256((output/'calibration.json').read_bytes()).hexdigest()==frozen
    write_json(output/'summary.json',{'experiment_id':'exp_42','trial_count':len(rows),'protocol':PROTOCOL,
               'groups':group_rows,'paired':paired,'pairing_checks':pairing,
               'protected_source_sha256':before,'protected_sources_unchanged':unchanged,
               'frozen_calibration_sha256':frozen,'calibration_unchanged':frozen_unchanged,
               'all_horizons_completed':all(r['required_horizon_completed'] for r in rows),
               'limitations':PROTOCOL['limitations']})
    for r in group_rows:
        if r['policy']=='adaptive':
            if r['fault_yaw_deg']==0:
                print(f"{r['environment']}/{r['normal_motion']}/healthy/adaptive: yanlış alarm={r['false_alarm']}/{r['trials']}",flush=True)
            else:
                print(f"{r['environment']}/{r['normal_motion']}/{r['fault_yaw_deg']:g}deg/adaptive: alarm={r['alarm_trials']}/{r['trials']} | doğru kaynak={r['correct_source']}/{r['trials']} | belirsiz={r['uncertain']} | yanlış={r['wrong_source']}",flush=True)
    print(f"Eski kaynaklar korunmuş: {'PASS' if unchanged else 'FAIL'}")
    print(f"Kalibrasyon değişmemiş: {'PASS' if frozen_unchanged else 'FAIL'}")
    print(f'Sonuç klasörü: {output}')
    if not unchanged or not frozen_unchanged:
        raise RuntimeError('Kaynak/kalibrasyon koruma kontrolü başarısız.')


if __name__=='__main__':
    main()
