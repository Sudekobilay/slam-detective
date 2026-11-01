"""Öğrenilmiş çizgi haritasından tanısal hareket seçen ilk geometrik model.

Girdi bir kestiricinin inanç durumudur; gerçek harita/konum veya hata etiketi
alınmaz. Çıktı yaklaşık bir beklenen ayırt edicilik skorudur, teşhis olasılığı
veya güvenlik garantisi değildir. Tek hata varsayımı kullanılır.
"""

import math
from itertools import combinations

import numpy as np

from estimation.line_ekf_slam import line_observation
from estimation.pose_ekf import motion_prediction
from simulator.mounting import SensorMount
from simulator.robot import Pose, wrap_angle

CANDIDATES = {'straight': (4.0, 4.0), 'arc': (3.1, 4.9)}
PARAMETERS = ('front_yaw', 'rear_yaw', 'heading_offset')


def _segment_distance(point, start, end):
    point, start, end = (np.asarray(value, dtype=float) for value in (point, start, end))
    direction = end - start
    squared = float(direction @ direction)
    fraction = float(np.clip((point - start) @ direction / squared, 0., 1.)) if squared > 1e-20 else 0.
    return float(np.linalg.norm(point - (start + fraction * direction)))


def _clearance(pose, features, radius):
    return min(_segment_distance((pose.x, pose.y), f.start, f.end) for f in features) - radius


def _line_of_sight(origin, target, features):
    origin, target = np.asarray(origin), np.asarray(target)
    direction = target - origin
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return False
    direction /= length
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    for feature in features:
        start, end = np.asarray(feature.start), np.asarray(feature.end)
        segment = end - start
        denom = cross(direction, segment)
        if abs(denom) < 1e-12:
            continue
        offset = start - origin
        distance = cross(offset, segment) / denom
        fraction = cross(offset, direction) / denom
        if 1e-6 < distance < length - 1e-5 and -1e-9 <= fraction <= 1 + 1e-9:
            return False
    return True


def _visible(feature, sensor_pose, features, field_of_view, max_range):
    samples = (feature.start, feature.end, (
        (feature.start[0] + feature.end[0]) / 2.,
        (feature.start[1] + feature.end[1]) / 2.,
    ))
    for point in samples:
        dx, dy = point[0] - sensor_pose.x, point[1] - sensor_pose.y
        distance = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - sensor_pose.theta)
        if (
            .05 < distance <= max_range
            and abs(angle) <= field_of_view / 2.
            and _line_of_sight((sensor_pose.x, sensor_pose.y), point, features)
        ):
            return True
    return False


def _separation_scores(information, probe_rad):
    """Alternatif hata büyüklüğü/işareti serbestken yerel altuzay ayrımı."""
    diagonal = np.diag(information)
    scores = {f'nominal_vs_{name}': float(probe_rad**2 * max(0., diagonal[i])) for i, name in enumerate(PARAMETERS)}
    for i, j in combinations(range(3), 2):
        if diagonal[i] <= 1e-15 or diagonal[j] <= 1e-15:
            separation = 0.
        else:
            separation = probe_rad**2 * max(0., min(
                diagonal[i] - information[i, j]**2 / diagonal[j],
                diagonal[j] - information[i, j]**2 / diagonal[i],
            ))
        scores[f'{PARAMETERS[i]}_vs_{PARAMETERS[j]}'] = float(separation)
    return scores


def select_diagnostic_motion(estimator):
    """İki adayı yalnız mevcut tahmin, kovaryans ve öğrenilmiş haritayla puanlar.

    Hipotezler küçük ön/arka montaj açısı ve yön hatalarıdır. Gelecekte
    görüneceği tahmin edilen çizgilerin duyarlılıkları karşılaştırılır.
    Bilinen harita parçalarına yakın adaylar elenir; bilinmeyen alanın
    çarpışmasız olduğu garanti edilmez. Adaylar eşit süre ve mesafededir.
    """
    state = np.asarray(estimator.state, dtype=float).copy()
    covariance = np.asarray(estimator.P, dtype=float).copy()
    noise = np.asarray(estimator.line_R, dtype=float).copy()
    features = tuple(estimator.map_features())
    if (
        state.ndim != 1 or len(state) != 4 + 2 * len(features)
        or covariance.shape != (len(state), len(state))
        or noise.shape != (2, 2)
        or not np.isfinite(state).all()
        or not np.isfinite(covariance).all()
        or not np.isfinite(noise).all()
        or not np.allclose(covariance, covariance.T, atol=1e-10)
        or np.linalg.eigvalsh(covariance).min() < -1e-8
        or not np.allclose(noise, noise.T, atol=1e-10)
        or np.linalg.eigvalsh(noise).min() <= 0.
    ):
        raise ValueError('Geçersiz planlama inancı.')
    result = {
        'status': 'no_supported_candidate', 'selected_motion': None,
        'selection_time_s': float(estimator.time_s),
        'map_line_count': len(features),
        'candidates': [],
        'model': 'local sensitivity subspace separation proxy',
        'duration_s': 2., 'dt_s': .1,
        'probe_angle_deg': 1.,
        'safety_model': 'known estimated line segments only',
    }
    if not features:
        result['reason'] = 'empty_estimated_map'
        return result
    mounts = {'front': SensorMount(x=.25), 'rear': SensorMount(x=-.25, yaw=math.pi)}
    geometry = estimator.geometry
    radius, margin = .35, .05
    for name, (left, right) in CANDIDATES.items():
        predicted = state.copy()
        P = covariance.copy()
        information = np.zeros((3, 3))
        sample_rows = []
        v = (geometry.left_radius * left + geometry.right_radius * right) / 2.
        omega = (geometry.right_radius * right - geometry.left_radius * left) / geometry.wheelbase
        clearance = _clearance(Pose(*predicted[:3]), features, radius)
        safe = clearance > margin
        for step in range(1, 21):
            if not safe:
                break
            # Beklenen gyro ölçümü = planlanan dönüş + mevcut bias tahmini.
            # Bu, gerçek gyro veya gerçek robot konumuna erişim değildir.
            robot, F_robot, G = motion_prediction(
                predicted[:4], v * .1, (omega + predicted[3]) * .1, .1
            )
            F = np.eye(len(predicted))
            F[:4, :4] = F_robot
            Q = G @ np.diag([
                estimator.distance_noise_density**2 * .1,
                .01**2 * .1**2,
            ]) @ G.T
            Q[3, 3] += estimator.bias_random_walk**2 * .1
            P = F @ P @ F.T
            P[:4, :4] += Q
            P = (P + P.T) / 2.
            predicted[:4] = robot
            pose = Pose(*predicted[:3])
            clearance = min(clearance, _clearance(pose, features, radius))
            # Küçük adımlar arasındaki merkez hareketini de ihtiyatla hesaba kat.
            if clearance <= margin + abs(v) * .1:
                safe = False
                break
            if step % 5:
                continue
            H_rows, J_rows = [], []
            counts = {'front': 0, 'rear': 0}
            for sensor, mount in mounts.items():
                sensor_pose = mount.world_pose(pose)
                for index, feature in enumerate(features):
                    if not _visible(feature, sensor_pose, features, math.pi, 15.):
                        continue
                    start = 4 + 2 * index
                    _, H_robot, H_landmark = line_observation(
                        predicted[:4], predicted[start:start+2], mount
                    )
                    H = np.zeros((2, len(predicted)))
                    H[:, :4], H[:, start:start+2] = H_robot, H_landmark
                    J = np.zeros((2, 3))
                    J[1, 0 if sensor == 'front' else 1] = -1.
                    J[:, 2] = H_robot[:, 2]
                    H_rows.append(H)
                    J_rows.append(J)
                    counts[sensor] += 1
            if H_rows:
                H, J = np.vstack(H_rows), np.vstack(J_rows)
                S = H @ P @ H.T + np.kron(np.eye(len(H_rows)), noise)
                S = (S + S.T) / 2.
                L = np.linalg.cholesky(S)
                whitened = np.linalg.solve(L, J)
                information += whitened.T @ whitened
            sample_rows.append({'relative_time_s': step * .1, 'expected_lines': counts})
        pair_scores = _separation_scores(information, math.radians(1.))
        score = min(pair_scores.values()) if safe else None
        supported = safe and score is not None and score > 1e-12
        result['candidates'].append({
            'motion': name, 'eligible': supported,
            'known_map_clearance_check': safe,
            'minimum_known_clearance_m': clearance,
            'predicted_separation_score': score,
            'pair_scores': pair_scores,
            'samples': sample_rows,
            'predicted_final_pose': [float(value) for value in predicted[:3]],
            'distance_m': abs(v) * 2., 'absolute_turn_rad': abs(omega) * 2.,
        })
    eligible = [row for row in result['candidates'] if row['eligible']]
    if eligible:
        best = max(row['predicted_separation_score'] for row in eligible)
        tied = [row for row in eligible if math.isclose(row['predicted_separation_score'], best, rel_tol=1e-9, abs_tol=1e-12)]
        chosen = min(tied, key=lambda row: (row['absolute_turn_rad'], row['motion']))
        result.update(status='selected', selected_motion=chosen['motion'])
    else:
        result['reason'] = 'insufficient_expected_separation_or_known_obstacle'
    return result
