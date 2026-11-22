"""Validated benchmark configuration, independent of execution."""
import json
import math
from pathlib import Path
from configuration.protocols import CALIBRATION_SEEDS
from estimation.diagnostic_motion import CANDIDATES
from simulator.validation_environments import validation_environment

def read_config(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    allowed = {'environments', 'normal_motions', 'starts', 'noise_profiles', 'fault_angles_deg', 'seeds'}
    if not isinstance(data, dict) or set(data) != allowed:
        raise ValueError('Benchmark fields are missing or unknown.')
    for key in ('environments','normal_motions','starts','noise_profiles','fault_angles_deg','seeds'):
        if not isinstance(data.get(key), list) or not data[key]:
            raise ValueError(f'Nonempty list required: {key}')
    if any(type(s) is not int or s < 0 for s in data['seeds']) or len(set(data['seeds'])) != len(data['seeds']):
        raise ValueError('Seeds must be distinct nonnegative integers.')
    if set(data['seeds']) & set(CALIBRATION_SEEDS):
        raise ValueError('Calibration and evaluation seeds must be separate.')
    for collection in ('starts','noise_profiles'):
        fields = {'id', 'x_m', 'y_m', 'yaw_deg'} if collection == 'starts' else {'id', 'range_std_m', 'dropout_probability'}
        if any(not isinstance(r, dict) or set(r) != fields for r in data[collection]):
            raise ValueError('Missing or unknown condition fields.')
        ids = [r['id'] for r in data[collection]]
        if any(not isinstance(x,str) or not x.replace('_','').isalnum() for x in ids) or len(ids) != len(set(ids)):
            raise ValueError('Unique simple identifiers required.')
    if any(not isinstance(x, str) for key in ('normal_motions', 'environments') for x in data[key]):
        raise ValueError('Environment and motion names must be strings.')
    for motion in data['normal_motions']:
        if motion not in CANDIDATES: raise ValueError('Invalid normal motion.')
    for world in data['environments']: validation_environment(world)
    for start in data['starts']:
        if not all(type(start[k]) in (int, float) and math.isfinite(start[k]) for k in ('x_m','y_m','yaw_deg')):
            raise ValueError('Start pose must be finite.')
    for profile in data['noise_profiles']:
        std,drop = profile['range_std_m'],profile['dropout_probability']
        if type(std) not in (int, float) or type(drop) not in (int, float) or not math.isfinite(std) or std < 0 or not math.isfinite(drop) or not 0 <= drop <= 1:
            raise ValueError('Invalid noise profile.')
    if any(type(a) not in (int, float) or not math.isfinite(a) or abs(a)>10 for a in data['fault_angles_deg']):
        raise ValueError('Fault angles must be finite, within +/-10 degrees.')
    for key in ('environments','normal_motions','fault_angles_deg'):
        if len(set(data[key]))!=len(data[key]): raise ValueError('Duplicate protocol conditions.')
    if 0. not in data['fault_angles_deg']: raise ValueError('Healthy control is required.')
    return data
