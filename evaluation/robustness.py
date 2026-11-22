"""Pure result evaluation; no CLI, world generation or filesystem writes."""
import math
import statistics
from evaluation.diagnosis import diagnosis_readout, outcome
from simulator.robot import wrap_angle

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
