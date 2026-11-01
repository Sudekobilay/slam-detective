"""Yapılandırılabilir LiDAR listesiyle geometrik izleme ve hareket seçimi.

Tek-hata modelleridir. Puanlar olasılık değildir; tek LiDAR'da kaynak
ayrımı desteklenmez. Karar girdilerinde gerçek harita/konum bulunmaz.
"""
import copy
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np

from estimation.diagnostic_motion import CANDIDATES, _clearance
from estimation.raw_scan_hypotheses import _cost_curve
from estimation.scan_alignment_guard import scan_alignment
from simulator.environment import Wall
from simulator.lidar import Lidar
from simulator.robot import DifferentialDriveRobot, Pose


def usable_alignments(alignments, config):
    return {
        sensor: row for sensor, row in alignments.items()
        if row is not None and not row['at_search_edge']
        and row['alignment_rms_m'] <= config.max_alignment_rms_m
    }


class LayoutScanGuard:
    """Her destekli sensör çifti için ayrı, işaret tutarlı alarm sayacı."""
    def __init__(self, config):
        self.config = config
        self.confirmed = self.quarantine_active = False
        self.history = []
        self.last_result = None
        self._last_time = None
        self._streaks = {}

    def observe(self, estimator, packets, mounts, time_s):
        if not math.isfinite(time_s) or time_s < 0 or (
            self._last_time is not None and time_s <= self._last_time
        ):
            raise ValueError('Geçersiz tarama zamanı.')
        alignments = {}
        for packet in packets:
            sensor = packet['sensor_id']
            if sensor not in mounts or sensor in alignments:
                raise ValueError('Tanımsız veya tekrarlanan LiDAR.')
            alignments[sensor] = scan_alignment(
                estimator, packet, mounts[sensor], self.config
            )
        usable = usable_alignments(alignments, self.config)
        sensors = sorted(usable)
        streaks, pairs = {}, []
        for index, first in enumerate(sensors):
            for second in sensors[index + 1:]:
                gap = usable[first]['offset_deg'] - usable[second]['offset_deg']
                warning = abs(gap) > self.config.relative_yaw_threshold_deg
                sign = 1 if gap > 0 else -1
                previous_sign, previous_count = self._streaks.get((first, second), (0, 0))
                count = (previous_count + 1 if previous_sign == sign else 1) if warning else 0
                streaks[first, second] = (sign, count)
                if count >= self.config.consecutive_scans:
                    self.confirmed = True
                pairs.append({'sensors': [first, second], 'gap_deg': float(gap),
                              'warning': bool(warning), 'streak': count})
        self._streaks = streaks
        warning = any(row['warning'] for row in pairs)
        if warning:
            self.quarantine_active = True
        elif len(usable) == len(mounts) and len(usable) >= 2 and not self.confirmed:
            self.quarantine_active = False
        status = ('confirmed' if self.confirmed else 'watch' if warning
                  else 'consistent' if pairs else 'insufficient_alignment_support')
        result = {'time_s': time_s, 'status': status, 'pairs': pairs,
                  'alignments': alignments, 'confirmed': self.confirmed,
                  'quarantine_active': self.quarantine_active}
        self._last_time, self.last_result = time_s, result
        self.history.append(result)
        return result


def compare_layout_hypotheses(estimator, packets, mounts, config):
    result = {'status': 'insufficient_support', 'nominal_cost_m2': None,
              'models': [], 'supported_sensors': [], 'unsupported_sensors': []}
    selected = {}
    seen = set()
    for packet in packets:
        sensor = packet['sensor_id']
        if sensor not in mounts or sensor in seen:
            raise ValueError('Tanımsız veya tekrarlanan LiDAR.')
        seen.add(sensor)
        points = np.asarray([p for p in packet['data']['scan']['points'] if p is not None], dtype=float)
        if points.size and (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()):
            raise ValueError('Geçersiz ham noktalar.')
        if len(points) >= max(20, config.min_points):
            selected[sensor] = points
    result['supported_sensors'] = sorted(selected)
    result['unsupported_sensors'] = sorted(set(mounts) - set(selected))
    if len(selected) < 2:
        result['reason'] = 'at_least_two_supported_lidars_required'
        return result
    features = tuple(estimator.map_features())
    if not features:
        return result
    starts = np.asarray([f.start for f in features], dtype=float)
    directions = np.asarray([f.end for f in features], dtype=float) - starts
    lengths = np.sum(directions ** 2, axis=1)
    valid = lengths > 1e-12
    starts, directions, lengths = starts[valid], directions[valid], lengths[valid]
    if not len(starts):
        return result
    count = math.floor(config.angle_limit_deg / config.angle_step_deg + 1e-12)
    angles = np.asarray(sorted((i * config.angle_step_deg for i in range(-count, count + 1)),
                               key=lambda value: (abs(value), value)))
    curves, common = {}, {}
    for sensor, points in selected.items():
        args = (points, mounts[sensor], estimator.pose, starts, directions, lengths, angles)
        curves[sensor] = _cost_curve(*args, False, config.trim_fraction)
        common[sensor] = _cost_curve(*args, True, config.trim_fraction)
    nominal = float(np.mean([curve[0] for curve in curves.values()]))
    result.update(status='evaluated', nominal_cost_m2=nominal)
    for model in [f'{sensor}_yaw' for sensor in sorted(selected)] + ['heading_offset']:
        per_sensor = [common[s] if model == 'heading_offset' else curves[s]
                      if model == f'{s}_yaw' else np.full(len(angles), curves[s][0])
                      for s in sorted(selected)]
        costs = np.mean(per_sensor, axis=0)
        best = int(np.argmin(costs))
        result['models'].append({
            'hypothesis': model, 'best_offset_deg': float(angles[best]),
            'cost_m2': float(costs[best]), 'improvement_m2': float(nominal - costs[best]),
            'max_sensor_rms_m': float(math.sqrt(max(curve[best] for curve in per_sensor))),
            'at_search_edge': bool(math.isclose(abs(angles[best]), max(abs(angles)))),
        })
    result['models'].sort(key=lambda row: row['cost_m2'])
    return result


def select_layout_motion(estimator, raw, mounts, lidar_configs):
    result = {'status': 'no_supported_candidate', 'selected_motion': None,
              'candidates': [], 'models': []}
    if raw['status'] != 'evaluated':
        return result
    models = raw['models'][:2]
    result['models'] = copy.deepcopy(models)
    features = tuple(estimator.map_features())
    if len(models) < 2 or not features:
        return result
    environment = SimpleNamespace(walls=tuple(Wall(f.start, f.end) for f in features))
    lidars = {sensor: Lidar(lidar_configs[sensor]) for sensor in mounts}
    sensors = raw['supported_sensors']
    for name, rates in CANDIDATES.items():
        predictions, safe = [], True
        for model in models:
            angle = math.radians(model['best_offset_deg'])
            pose = estimator.pose
            if model['hypothesis'] == 'heading_offset':
                pose = Pose(pose.x, pose.y, pose.theta + angle)
            robot = DifferentialDriveRobot(estimator.geometry, pose)
            hypothetical = {
                sensor: replace(mount, yaw=mount.yaw + angle)
                if model['hypothesis'] == f'{sensor}_yaw' else mount
                for sensor, mount in mounts.items()
            }
            speed = (estimator.geometry.left_radius * rates[0]
                     + estimator.geometry.right_radius * rates[1]) / 2
            margin = .05 + abs(speed) * .1
            safe = safe and _clearance(robot.pose, features, .35) > margin
            scans = []
            for step in range(1, 21):
                robot.step(*rates, .1)
                safe = safe and _clearance(robot.pose, features, .35) > margin
                if step % 5 == 0:
                    scans.append({sensor: lidars[sensor].scan(
                        hypothetical[sensor].world_pose(robot.pose), environment
                    ).ranges for sensor in sensors})
            predictions.append(scans)
        scores, counts = [], []
        for first, second in zip(*predictions):
            sensor_scores, frame_counts = [], {}
            for sensor in sensors:
                values = [min(abs(a - b), .5) ** 2
                          for a, b in zip(first[sensor], second[sensor])
                          if a is not None and b is not None]
                frame_counts[sensor] = len(values)
                if len(values) >= 20:
                    sensor_scores.append(sum(values) / len(values))
            counts.append(frame_counts)
            if len(sensor_scores) >= 2:
                scores.append(sum(sensor_scores) / len(sensor_scores))
        score = sum(scores) / 4 if len(scores) == 4 else None
        result['candidates'].append({
            'motion': name, 'eligible': bool(safe and score is not None and score > 1e-12),
            'known_map_clearance': bool(safe), 'score_m2': score,
            'common_ray_counts': counts,
        })
    eligible = [row for row in result['candidates'] if row['eligible']]
    if eligible:
        chosen = max(eligible, key=lambda row: (row['score_m2'], row['motion'] == 'straight'))
        result.update(status='selected', selected_motion=chosen['motion'])
    return result
