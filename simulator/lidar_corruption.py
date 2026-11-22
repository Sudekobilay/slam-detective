"""Keyed radial noise/dropout; identical sensor/time/beam uses identical draws."""
from copy import deepcopy
import hashlib
import math
import numpy as np


def corrupt_lidar_event(event, *, seed, range_std_m=0., dropout_probability=0., max_range_m=15.):
    if not math.isfinite(range_std_m) or range_std_m < 0:
        raise ValueError('Range standard deviation must be finite and nonnegative.')
    if not math.isfinite(dropout_probability) or not 0 <= dropout_probability <= 1:
        raise ValueError('Dropout probability must be in [0,1].')
    if not math.isfinite(max_range_m) or max_range_m <= 0:
        raise ValueError('Range must be positive.')
    if event['sensor_type'] != 'lidar':
        raise ValueError('LiDAR event required.')
    result = deepcopy(event)
    if range_std_m == 0 and dropout_probability == 0:
        return result
    scan = result['data']['scan']
    if len(scan['angles']) != len(scan['ranges']) or len(scan['points']) != len(scan['ranges']):
        raise ValueError('Inconsistent scan dimensions.')
    key = f"{seed}|{event['sensor_id']}|{event['time_s']:.12f}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
    drops = rng.random(len(scan['ranges']))
    noise = rng.normal(0., range_std_m, len(scan['ranges']))
    ranges, points = [], []
    for angle, distance, drop, epsilon in zip(scan['angles'], scan['ranges'], drops, noise):
        if distance is None or drop < dropout_probability:
            ranges.append(None); points.append(None); continue
        value = float(distance + epsilon)
        if not 0 < value <= max_range_m:
            ranges.append(None); points.append(None); continue
        ranges.append(value)
        points.append([value*math.cos(angle), value*math.sin(angle)])
    scan['ranges'], scan['points'] = ranges, points
    return result
