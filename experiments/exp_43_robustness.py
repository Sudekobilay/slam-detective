"""Held-out poses and radial LiDAR noise, frozen EXP-42 calibration, four policies."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from estimation.diagnostic_motion import CANDIDATES
from estimation.line_ekf_slam import LineEKFSLAM
from experiments.exp_24_motion_comparison import write_json, write_jsonl
from experiments.exp_36_sensor_layouts import examples
from experiments.exp_40_configurable_active import LayoutWorld, create_agent
from experiments.exp_42_heldout_environments import (
    CALIBRATION_SEEDS, POLICIES, calibrate_readout, check_pairing, diagnosis_readout,
    outcome, protected_hashes,
)
from experiments.line_slam_common import write_csv
from simulator.collision import try_step
from simulator.lidar_corruption import corrupt_lidar_event
from simulator.robot import DifferentialDriveRobot, Pose, wrap_angle
from simulator.validation_environments import validation_environment


def read_config(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key in ('environments','normal_motions','starts','noise_profiles','fault_angles_deg','seeds'):
        if not isinstance(data.get(key), list) or not data[key]:
            raise ValueError(f'Nonempty list required: {key}')
    if len(set(data['seeds'])) != len(data['seeds']) or any(type(s) is not int or s < 0 for s in data['seeds']):
        raise ValueError('Seeds must be distinct nonnegative integers.')
    if set(data['seeds']) & set(CALIBRATION_SEEDS):
        raise ValueError('Calibration and evaluation seeds must be separate.')
    for collection in ('starts','noise_profiles'):
        ids = [r['id'] for r in data[collection]]
        if len(ids) != len(set(ids)) or any(not isinstance(x,str) or not x.replace('_','').isalnum() for x in ids):
            raise ValueError('Unique simple identifiers required.')
    for motion in data['normal_motions']:
        if motion not in CANDIDATES: raise ValueError('Invalid normal motion.')
    for world in data['environments']: validation_environment(world)
    for start in data['starts']:
        if not all(math.isfinite(start[k]) for k in ('x_m','y_m','yaw_deg')):
            raise ValueError('Start pose must be finite.')
    for profile in data['noise_profiles']:
        std,drop = profile['range_std_m'],profile['dropout_probability']
        if not math.isfinite(std) or std < 0 or not math.isfinite(drop) or not 0 <= drop <= 1:
            raise ValueError('Invalid noise profile.')
    if any(not math.isfinite(a) or abs(a)>10 for a in data['fault_angles_deg']):
        raise ValueError('Fault angles must be finite, within +/-10 degrees.')
    for key in ('environments','normal_motions','fault_angles_deg'):
        if len(set(data[key]))!=len(data[key]): raise ValueError('Duplicate protocol conditions.')
    if 0. not in data['fault_angles_deg']: raise ValueError('Healthy control is required.')
    return data


def pose_of(start):
    return Pose(start['x_m'],start['y_m'],math.radians(start['yaw_deg']))


class RobustnessWorld(LayoutWorld):
    def __init__(self,data,environment,start,profile,seed,yaw):
        super().__init__(data,seed,'healthy' if yaw==0 else 'lidar','front',yaw)
        self._environment = validation_environment(environment)
        self._robot = DifferentialDriveRobot(initial_pose=pose_of(start))
        self.profile,self.noise_seed = dict(profile),seed+900000
        self.start = dict(start)

    def _batch(self,motion_packets):
        events = super()._batch(motion_packets)
        configs = {u.sensor_id:u.config for u in self.units}
        return [corrupt_lidar_event(e,seed=self.noise_seed,
                range_std_m=self.profile['range_std_m'],
                dropout_probability=self.profile['dropout_probability'],
                max_range_m=configs[e['sensor_id']].max_range)
                if e['sensor_type']=='lidar' else e for e in events]


def preflight(environment,start,normal):
    world = validation_environment(environment)
    robot = DifferentialDriveRobot(initial_pose=pose_of(start))
    poses = [robot.pose]
    # Check footprint at t=0 as well as each step and every candidate branch.
    if not try_step(robot,world,0.,0.,.1,radius=.35): return False
    for _ in range(40):
        if not try_step(robot,world,*CANDIDATES[normal],.1,radius=.35):return False
        poses.append(robot.pose)
    for pose in poses:
        for rates in CANDIDATES.values():
            branch = DifferentialDriveRobot(initial_pose=pose)
            for _ in range(20):
                if not try_step(branch,world,*rates,.1,radius=.35):return False
    return True


def run_trial(data,config,environment,start,profile,seed,yaw,normal,policy):
    world = RobustnessWorld(data,environment,start,profile,seed,yaw)
    agent = create_agent(data,config,'ekf',policy=policy,normal=normal,seed=seed+20000,use_gps=False)
    # Identical known initial prior is a condition of this experiment, not GT feedback.
    agent.reference.estimator = LineEKFSLAM(initial_pose=pose_of(start))
    events = world.initial_measurements();agent.process(events)
    commands,reason,deadline = [],None,4.
    tick = time.perf_counter()
    while agent.time_s < deadline-1e-9:
        rates = agent.command()
        cmd = {'interval_start_s':agent.time_s,'left_rate_rad_s':rates[0],
               'right_rate_rad_s':rates[1],'mode':agent.mode,'applied':False,
               'changed_from_normal':rates != agent.normal_command}
        commands.append(cmd)
        try:batch = world.step(*rates)
        except RuntimeError as e:reason=str(e);break
        cmd['applied']=True;agent.process(batch);events.extend(batch)
        deadline=max(4.,agent.window_end_s or 4.)
    return agent,events,commands,world.evaluation_records(),reason,time.perf_counter()-tick


def path_cost(truth,start,end):
    selected = [r for r in truth if start-1e-9 <= r['time_s'] <= end+1e-9]
    return {
        'distance_m':sum(math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m']) for a,b in zip(selected,selected[1:])),
        'absolute_turn_rad':sum(abs(wrap_angle(b['theta_rad']-a['theta_rad'])) for a,b in zip(selected,selected[1:])),
        'duration_s':max(0.,selected[-1]['time_s']-selected[0]['time_s']) if selected else 0.,
    }


def trial_row(trial,thresholds,max_rms,case):
    agent,events,commands,records,reason,elapsed = trial
    complete = reason is None and agent.time_s >= max(4.,agent.window_end_s or 4.)-1e-9
    window_complete = agent.trigger_time_s is not None and agent.time_s >= agent.window_end_s-1e-9
    readout = diagnosis_readout(agent.raw_history,agent.trigger_time_s,agent.window_end_s,thresholds,max_rms)
    truth = {round(r['time_s'],9):r for r in records['truth']}
    errors = [(p['x_m']-truth[round(p['time_s'],9)]['x_m'])**2+
              (p['y_m']-truth[round(p['time_s'],9)]['y_m'])**2
              for p in agent.path if p['time_s']<=4.+1e-9]
    costs = path_cost(records['truth'],0.,min(4.,agent.time_s))
    confirmation = readout['confirmation_time_s']
    # Costs use a common 0..4s horizon; decision time is separate. No fabricated extra distance.
    return {**case,'trigger_time_s':agent.trigger_time_s,'selected_motion':agent.selected_motion,
            'required_horizon_completed':complete,'stop_reason':reason,
            'outcome':outcome(case['fault_yaw_deg'],agent.trigger_time_s,complete and window_complete,readout['confirmed_model']),
            **readout,'common_position_rmse_m':math.sqrt(statistics.mean(errors)) if agent.time_s>=4.-1e-9 else None,
            **{'common_'+k:v for k,v in costs.items()},
            'confirmation_delay_from_alarm_s':confirmation-agent.trigger_time_s if confirmation is not None else None,
            'confirmation_delay_from_fault_s':confirmation-2. if confirmation is not None else None,
            'changed_command_duration_s':sum(.1 for c in commands if c['applied'] and c['changed_from_normal']),
            'local_loop_runtime_s':elapsed}


def paired_rows(rows):
    groups = {}
    for r in rows:
        key=tuple(r[k] for k in ('environment','start_id','noise_id','normal_motion','seed','fault_yaw_deg'))
        groups.setdefault(key,{})[r['policy']]=r
    out=[]
    for policies in groups.values():
        a=policies['adaptive']
        for comparator in ('passive','fixed_straight','fixed_arc'):
            b=policies[comparator]
            row={k:a[k] for k in ('environment','start_id','noise_id','normal_motion','seed','fault_yaw_deg')}
            row.update(comparator=comparator,adaptive_outcome=a['outcome'],comparator_outcome=b['outcome'],
                       correct_source_difference=int(a['outcome']=='correct_source')-int(b['outcome']=='correct_source'))
            for metric in ('common_distance_m','common_absolute_turn_rad','common_position_rmse_m','confirmation_delay_from_fault_s'):
                row['delta_'+metric]=a[metric]-b[metric] if a[metric] is not None and b[metric] is not None else None
            out.append(row)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/benchmarks/robustness.json'))
    parser.add_argument('--smoke',action='store_true',help='24 runs: nominal pose, corridor, arc, one seed; not the full study.')
    parser.add_argument('--save-recordings',action='store_true')
    args=parser.parse_args();cfg=read_config(args.config)
    if args.smoke:
        cfg={**cfg,'environments':['corridor'],'normal_motions':['arc'],
             'starts':[cfg['starts'][0]],'seeds':[cfg['seeds'][0]]}
    root=Path(__file__).resolve().parents[1];before=protected_hashes(root)
    out=root/'results'/('exp_43_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
    checks={f'{env}/{s["id"]}/{n}':preflight(env,s,n) for env in cfg['environments'] for s in cfg['starts'] for n in cfg['normal_motions']}
    protocol={**cfg,'experiment_id':'exp_43','policies':list(POLICIES),'smoke_only':args.smoke,
              'algorithm':'ekf','layout':'quad','gps_enabled':False,'calibration':'EXP-42 healthy original environment, ideal LiDAR; no retuning',
              'known_initial_pose_prior':True,'lidar_model':'independent Gaussian radial range noise + Bernoulli dropout; no angular noise/outliers/skew',
              'environment_geometry':{e:asdict(validation_environment(e)) for e in cfg['environments']},
              'common_cost_horizon_s':[0.,4.],'no_posthoc_tuning':True,'preflight':checks}
    write_json(out/'protocol.json',protocol)
    if not all(checks.values()):raise RuntimeError('Unsafe preflight condition; protocol not executed.')
    data={**examples()['quad'],'gps_enabled':False}
    config,thresholds,max_rms,calibration=calibrate_readout(data)
    write_json(out/'calibration.json',calibration);frozen=hashlib.sha256((out/'calibration.json').read_bytes()).hexdigest()
    rows=[];pairing={}
    for env in cfg['environments']:
      for start in cfg['starts']:
       for profile in cfg['noise_profiles']:
        for normal in cfg['normal_motions']:
         for seed in cfg['seeds']:
          for yaw in cfg['fault_angles_deg']:
            trials={p:run_trial(data,config,env,start,profile,seed,yaw,normal,p) for p in POLICIES}
            key=f'{env}/{start["id"]}/{profile["id"]}/{normal}/{seed}/{yaw:g}'
            pairing[key]=check_pairing({p:t[:5] for p,t in trials.items()})
            if not all(pairing[key].values()):raise RuntimeError('Paired histories differ: '+key)
            for policy,t in trials.items():
                case={'environment':env,'start_id':start['id'],'noise_id':profile['id'],
                      'normal_motion':normal,'seed':seed,'fault_yaw_deg':yaw,'policy':policy}
                row=trial_row(t,thresholds,max_rms,case);rows.append(row)
                folder=out/env/start['id']/profile['id']/normal/f'seed_{seed}'/f'yaw_{yaw:g}'/policy
                folder.mkdir(parents=True)
                write_json(folder/'summary.json',row);write_json(folder/'decision.json',t[0].decision)
                write_csv(folder/'ground_truth.csv',t[3]['truth']);write_csv(folder/'estimate.csv',t[0].path)
                write_csv(folder/'commands.csv',t[2]);write_jsonl(folder/'raw_hypotheses.jsonl',t[0].raw_history)
                write_csv(folder/'agent_history.csv',t[0].history)
                if args.save_recordings:write_jsonl(folder/'measurements.jsonl',t[1])
          write_csv(out/'trials.csv',rows)
          print(f'{env}/{start["id"]}/{profile["id"]}/{normal}/seed_{seed}: {len(rows)} runs',flush=True)
    write_csv(out/'paired.csv',paired_rows(rows))
    unchanged=protected_hashes(root)==before
    frozen_ok=hashlib.sha256((out/'calibration.json').read_bytes()).hexdigest()==frozen
    summary={'experiment_id':'exp_43','trial_count':len(rows),'paired_conditions':len(pairing),
             'all_pairings_passed':all(all(c.values()) for c in pairing.values()),
             'all_horizons_completed':all(r['required_horizon_completed'] for r in rows),
             'protected_sources_unchanged':unchanged,'calibration_unchanged':frozen_ok,
             'protected_source_sha256':before,'calibration_sha256':frozen,
             'limitations':['Known correct start prior; not initial localization uncertainty.',
             'Thresholds/model covariance calibrated with ideal original environment; noise shift is a stress test, not noise-calibrated performance.',
             'Three seeds per full cell; descriptive results, no significance/universal superiority claim.',
             'Equal-speed 2s maneuvers can have equal distance/time; cost difference may be zero.']}
    write_json(out/'summary.json',summary);print('Output:',out)
    if not unchanged or not frozen_ok:raise RuntimeError('Source/calibration integrity failed.')


if __name__=='__main__':main()
