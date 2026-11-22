"""Compose the robustness protocol; CLI, storage and evaluation have explicit boundaries."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import product
from pathlib import Path

from application.calibration import calibrate_readout
from application.layout_service import create_agent
from application.runner import run_episode
from configuration.benchmarks import read_config
from configuration.layouts import examples
from configuration.protocols import POLICIES
from evaluation.diagnosis import check_pairing
from evaluation.robustness import path_cost, trial_row, paired_rows
from simulator.robustness_world import pose_of, RobustnessWorld, preflight
from simulator.validation_environments import validation_environment
from storage.files import write_json, write_jsonl, write_csv
from storage.provenance import protected_hashes
from storage.runs import RunSession


def run_trial(data,config,environment,start,profile,seed,yaw,normal,policy):
    world = RobustnessWorld(data,environment,start,profile,seed,yaw)
    agent = create_agent(data,config,'ekf',policy=policy,normal=normal,seed=seed+20000,use_gps=False, initial_pose=pose_of(start))
    # Identical known initial prior is a condition of this experiment, not GT feedback.
    result = run_episode(world, agent)
    return (*result.legacy(), result.runtime_s)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/benchmarks/robustness.json')
    parser.add_argument('--smoke',action='store_true',help='24 runs: nominal pose, corridor, arc, one seed; not the full study.')
    parser.add_argument('--save-recordings',action='store_true')
    parser.add_argument('--output', type=Path, help='New run directory; must not exist.')
    parser.add_argument('--allow-partial', action='store_true', help='Keep incomplete scientific runs with exit 0; manifest remains PARTIAL.')
    args=parser.parse_args(argv);cfg=read_config(args.config)
    if args.smoke:
        cfg={**cfg,'environments':['corridor'],'normal_motions':['arc'],
             'starts':[cfg['starts'][0]],'seeds':[cfg['seeds'][0]]}
    root=Path(__file__).resolve().parents[1];before=protected_hashes(root)
    base = Path.cwd() / 'results'
    out = args.output.resolve() if args.output else base/('exp_43_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    with RunSession(out, root, {**cfg, 'smoke_only': args.smoke}, experiment_id='exp_43') as session:
        return execute_protocol(cfg, args, root, out, before, session)


def execute_protocol(cfg, args, root, out, before, session):
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
    conditions = product(cfg['environments'], cfg['starts'], cfg['noise_profiles'],
                         cfg['normal_motions'], cfg['seeds'], cfg['fault_angles_deg'])
    for env, start, profile, normal, seed, yaw in conditions:
        trials = {policy: run_trial(data, config, env, start, profile, seed, yaw, normal, policy)
                  for policy in POLICIES}
        key = f'{env}/{start["id"]}/{profile["id"]}/{normal}/{seed}/{yaw:g}'
        pairing[key] = check_pairing({policy: trial[:5] for policy, trial in trials.items()})
        if not all(pairing[key].values()):
            raise RuntimeError('Paired histories differ: ' + key)
        for policy, trial in trials.items():
            case = {'environment': env, 'start_id': start['id'], 'noise_id': profile['id'],
                    'normal_motion': normal, 'seed': seed, 'fault_yaw_deg': yaw, 'policy': policy}
            row = trial_row(trial, thresholds, max_rms, case)
            rows.append(row)
            save_trial(out, case, trial, row, save_recordings=args.save_recordings)
        write_csv(out / 'trials.csv', rows)
        session.checkpoint(len(rows))
        print(f'{key}: {len(rows)} runs', flush=True)
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
    session.finish(complete=summary['all_horizons_completed'])
    if not summary['all_horizons_completed'] and not args.allow_partial:
        raise RuntimeError('Incomplete horizons; inspect manifest and trial outcomes.')


def save_trial(output, case, trial, row, *, save_recordings=False):
    folder = (output / case['environment'] / case['start_id'] / case['noise_id'] /
              case['normal_motion'] / f"seed_{case['seed']}" / f"yaw_{case['fault_yaw_deg']:g}" / case['policy'])
    folder.mkdir(parents=True)
    agent, events, commands, records, _, _ = trial
    write_json(folder / 'summary.json', row)
    write_json(folder / 'decision.json', agent.decision)
    write_csv(folder / 'ground_truth.csv', records['truth'])
    write_csv(folder / 'estimate.csv', agent.path)
    write_csv(folder / 'commands.csv', commands)
    write_jsonl(folder / 'raw_hypotheses.jsonl', agent.raw_history)
    write_csv(folder / 'agent_history.csv', agent.history)
    if save_recordings:
        write_jsonl(folder / 'measurements.jsonl', events)
    # Written last: only this file declares the case complete on disk.
    write_json(folder / 'case_manifest.json', {'schema_version': 1, 'status': 'COMPLETED',
               'required_horizon_completed': row['required_horizon_completed'],
               'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in folder.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
