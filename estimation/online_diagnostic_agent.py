"""Yalnız ölçüm paketleriyle çalışan, NIS tetiklemeli ilk çevrim içi denetleyici.

GPS kaydedilir, bu deneyde güncellemeye alınmaz. Tetikleme sensör arızasının
kesin teşhisi değildir. LiDAR kilidi otomatik açılmaz.
"""
from collections import Counter
import math

from estimation.diagnostic_motion import CANDIDATES, select_diagnostic_motion
from estimation.lidar_update_hold import LidarUpdateHold
from estimation.line_ekf_slam import LineEKFSLAM
from estimation.mounting_hypotheses import compare_mounting_hypotheses
from perception.line_association import associate_lines, transform_line
from perception.line_features import extract_lines
from simulator.encoder import EncoderMeasurement
from simulator.imu import GyroMeasurement
from simulator.mounting import SensorMount
from simulator.robot import Pose


class OnlineDiagnosticAgent:
    normal_command = (3.1, 4.9)

    def __init__(self, nis_threshold=9.21034, strategy='adaptive', scan_guard=None):
        if strategy not in ('adaptive', 'straight', 'passive'):
            raise ValueError('Geçersiz teşhis stratejisi.')
        self.strategy = strategy
        self.estimator = LineEKFSLAM(initial_pose=Pose(1., 3., 0.))
        self.scan_guard = scan_guard
        if scan_guard is None:
            self.hold = LidarUpdateHold(nis_threshold)
        else:
            from estimation.scan_alignment_guard import CombinedLidarGate
            self.hold = CombinedLidarGate(nis_threshold, scan_guard)
        self.mounts = {'front': SensorMount(x=.25), 'rear': SensorMount(x=-.25, yaw=math.pi)}
        self.mode = 'NORMAL'
        self.decision = None
        self.selected_motion = None
        self.window_end_s = None
        self._last_batch_time = None
        self.path, self.innovations, self.hypotheses, self.history = [], [], [], []
        self.counts = Counter()

    def command(self):
        """Bir sonraki aralık için komut; gerçek dünya nesnesi almaz."""
        if not self.hold.active:
            return self.normal_command
        if self.mode == 'NO_SUPPORTED_MANEUVER':
            return (0., 0.)
        if self.estimator.time_s >= self.window_end_s - 1e-9:
            return (0., 0.)
        if self.strategy == 'passive':
            return self.normal_command
        return CANDIDATES[self.selected_motion]

    def process(self, events):
        if not events:
            raise ValueError('Boş ölçüm paketi grubu.')
        time_s = float(events[0]['time_s'])
        if (
            not math.isfinite(time_s) or time_s < 0
            or any(float(event['time_s']) != time_s or event['schema_version'] != 1 for event in events)
            or (self._last_batch_time is not None and time_s <= self._last_batch_time)
        ):
            raise ValueError('Geçersiz, karışık veya tekrarlanan paket zamanı.')
        keys = [(event['sensor_type'], event['sensor_id']) for event in events]
        if len(set(keys)) != len(keys):
            raise ValueError('Aynı anda tekrarlanan sensör paketi.')
        encoders = [event for event in events if event['sensor_type'] == 'encoder']
        gyros = [event for event in events if event['sensor_type'] == 'gyro']
        if encoders or gyros:
            if len(encoders) != 1 or len(gyros) != 1:
                raise ValueError('Encoder ve gyro eşlenemedi.')
            self.estimator.predict(
                EncoderMeasurement(time_s=time_s, **encoders[0]['data']),
                GyroMeasurement(time_s=time_s, **gyros[0]['data']),
            )
        if not math.isclose(self.estimator.time_s, time_s, rel_tol=0., abs_tol=1e-9):
            raise ValueError('Ölçüm ve kestirim zamanı uyuşmuyor.')
        packets = sorted(
            (event for event in events if event['sensor_type'] == 'lidar'),
            key=lambda event: event['sensor_id'],
        )
        cache, evidence, observations = {}, [], []
        references = self.estimator.map_features()
        for packet in packets:
            sensor = packet['sensor_id']
            if sensor not in self.mounts:
                raise ValueError('Tanımsız LiDAR.')
            mount = self.mounts[sensor]
            features = extract_lines(packet['data']['scan']['points'])
            cache[sensor] = features
            world = [transform_line(f, mount.world_pose(self.estimator.pose)) for f in features]
            for match in associate_lines(world, references):
                innovation = None
                if match['status'] == 'matched':
                    feature = features[match['observation_index']]
                    innovation = self.estimator.line_innovation(match['reference_index'], feature, mount)
                    evidence.append({'sensor_id': sensor, 'observation_index': match['observation_index'], 'nis': innovation['nis']})
                    observations.append((sensor, match['reference_index'], feature, mount))
                self.innovations.append({
                    'time_s': time_s, 'sensor_id': sensor, **match,
                    'innovation': innovation,
                })
        if packets:
            if self.scan_guard is not None:
                self.scan_guard.observe(self.estimator, packets, self.mounts, time_s)
            self.hypotheses.append({'time_s': time_s, **compare_mounting_hypotheses(self.estimator, observations)})
            just_triggered = self.hold.observe(time_s, evidence)
            if just_triggered:
                self.window_end_s = time_s + 2.
                if self.strategy == 'adaptive':
                    self.decision = select_diagnostic_motion(self.estimator)
                    self.selected_motion = self.decision['selected_motion']
                    self.mode = 'DIAGNOSING' if self.selected_motion else 'NO_SUPPORTED_MANEUVER'
                elif self.strategy == 'straight':
                    self.selected_motion = 'straight'
                    self.mode = 'DIAGNOSING'
                else:
                    self.selected_motion = None
                    self.mode = 'PASSIVE_HOLD'

        if not self.hold.active and self.scan_guard is not None:
            self.mode = 'WATCH' if self.scan_guard.quarantine_active else 'NORMAL'

        # Tetkik amaçlı ölçümler dahil, tetiklenen tarama LiDAR düzeltmesine alınmaz.
        for packet in packets:
            sensor = packet['sensor_id']
            features = cache[sensor]
            self.counts['lidar_scans'] += 1
            if self.hold.active:
                self.counts['held_scans'] += 1
                continue
            if self.scan_guard is not None and self.scan_guard.quarantine_active:
                self.counts['quarantined_scans'] += 1
                continue
            mount = self.mounts[sensor]
            world = [transform_line(f, mount.world_pose(self.estimator.pose)) for f in features]
            matches = associate_lines(world, self.estimator.map_features())
            for match in matches:
                feature = features[match['observation_index']]
                if match['status'] == 'matched':
                    self.estimator.update_line(match['reference_index'], feature, mount)
                elif match['status'] == 'no_candidate':
                    self.estimator.add_line(feature, mount)
        if self.hold.active and self.window_end_s is not None and time_s >= self.window_end_s - 1e-9:
            self.mode = 'DIAGNOSIS_WINDOW_ENDED'
        self._last_batch_time = time_s
        pose = self.estimator.pose
        self.path.append({'time_s': time_s, 'x_m': pose.x, 'y_m': pose.y, 'theta_rad': pose.theta})
        self.history.append({
            'time_s': time_s, 'mode': self.mode,
            'hold_active': self.hold.active,
            'landmark_count': self.estimator.landmark_count,
            'line_updates': self.estimator.line_updates,
        })
