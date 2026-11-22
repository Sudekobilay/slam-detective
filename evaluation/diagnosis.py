"""Offline diagnosis readout and pairing; fault labels never enter control."""
import statistics
from estimation.raw_scan_hypotheses import diagnose_raw

def check_pairing(trials):
    first = next(iter(trials.values()))
    trigger = first[0].trigger_time_s
    cutoff = trigger if trigger is not None else 4.
    prefix = [e for e in first[1] if e['time_s']<=cutoff+1e-9]
    truth = [r for r in first[3]['truth'] if r['time_s']<=cutoff+1e-9]
    return {
        'same_alarm':all(t[0].trigger_time_s==trigger for t in trials.values()),
        'same_window':all(t[0].window_end_s==first[0].window_end_s for t in trials.values()),
        'same_measurements_through_alarm':all([e for e in t[1] if e['time_s']<=cutoff+1e-9]==prefix for t in trials.values()),
        'same_truth_through_alarm':all([r for r in t[3]['truth'] if r['time_s']<=cutoff+1e-9]==truth for t in trials.values()),
    }


def diagnosis_readout(history,trigger,end,thresholds,max_rms):
    if trigger is None:
        return {'confirmed_model':None,'confirmation_time_s':None,'window_frames':0,
                'supported_frames':0,'mean_true_source_margin_m2':None}
    frames = [r for r in history if trigger+1e-9<r['time_s']<=end+1e-9]
    previous,count,confirmed,confirmation = None,0,None,None
    margins=[]
    for frame in frames:
        decision = diagnose_raw(frame,thresholds,max_rms)
        if frame['status']=='evaluated':
            costs = {m['hypothesis']:m['cost_m2'] for m in frame['models']}
            # True-source margin is an offline metric; never fed to the controller.
            if 'front_yaw' in costs:
                margins.append(min(v for k,v in costs.items() if k!='front_yaw')-costs['front_yaw'])
        if decision['status']=='candidate':
            identity = (decision['hypothesis'],1 if decision['offset_deg']>0 else -1)
            count = count+1 if identity==previous else 1
            previous = identity
            if confirmed is None and count>=2:
                confirmed,confirmation = identity[0],frame['time_s']
        else:
            count,previous = 0,None
    return {'confirmed_model':confirmed,'confirmation_time_s':confirmation,
            'window_frames':len(frames),'supported_frames':sum(r['status']=='evaluated' for r in frames),
            'mean_true_source_margin_m2':statistics.mean(margins) if len(margins)==4 else None}


def outcome(yaw,trigger,complete,model):
    if yaw==0:
        return 'healthy_no_alarm' if trigger is None else 'false_alarm'
    if trigger is None:
        return 'no_alarm'
    if trigger<2.-1e-9:
        return 'premature_alarm'
    if not complete:
        return 'incomplete_window'
    if model is None:
        return 'uncertain'
    return 'correct_source' if model=='front_yaw' else 'wrong_source'
