"""Üç SLAM çekirdeği için genişletilebilir çevrim içi denetleyici.

Kararlar GPS almayan bağımsız haritalamadan gelir. İsteğe bağlı ikinci
kestirici korumalı GPS füzyonu yapar; denetim referansını değiştirmez.
"""
import math
from types import SimpleNamespace
import numpy as np

from estimation.diagnostic_motion import CANDIDATES
from estimation.fastslam1 import FastSLAM1Backend
from estimation.fastslam2 import FastSLAM2Backend
from estimation.gps_consistency import GPSConsistencyMonitor,joint_status
from estimation.hypothesis_motion import select_hypothesis_motion
from estimation.raw_scan_hypotheses import compare_raw_hypotheses
from estimation.scan_alignment_guard import ScanAlignmentGuard
from estimation.slam_backend import LineSlamBackend
from simulator.encoder import EncoderMeasurement
from simulator.gps import GPSMeasurement
from simulator.imu import GyroMeasurement
from simulator.mounting import SensorMount
from simulator.robot import Pose,wrap_angle


MOUNTS = {'front':SensorMount(x=.25),'rear':SensorMount(x=-.25,yaw=math.pi)}


def make_backend(algorithm,particles=40,seed=901,mounts=None):
    mounts = MOUNTS if mounts is None else mounts
    if algorithm=='ekf':
        return LineSlamBackend(mounts)
    classes = {'fastslam1':FastSLAM1Backend,'fastslam2':FastSLAM2Backend}
    if algorithm not in classes:
        raise ValueError('Geçersiz algoritma.')
    return classes[algorithm](mounts,particle_count=particles,seed=seed)


def geometry_view(backend):
    estimator = backend.estimator
    if isinstance(backend,FastSLAM1Backend):
        best = backend.particles[int(np.argmax(backend.weights))]
        pose = Pose(*best.state[:3])
        features = [line.feature() for line in best.landmarks]
    else:
        pose,features = estimator.pose,estimator.map_features()
    return SimpleNamespace(pose=pose,geometry=estimator.geometry,map_features=lambda:features)


def gps_reference(backend):
    estimator = backend.estimator
    if isinstance(backend,FastSLAM1Backend):
        states = np.asarray([p.state for p in backend.particles])
        weights = backend.weights
        pose = backend.pose
        mean = np.array([pose.x,pose.y,pose.theta,float(weights@states[:,3])])
        delta = states-mean
        delta[:,2] = [wrap_angle(value) for value in delta[:,2]]
        covariance = delta.T@(delta*weights[:,None])+np.eye(4)*1e-10
        return SimpleNamespace(state=mean,P=covariance,time_s=backend.time_s)
    return estimator


class ActivePipeline:
    def __init__(self,algorithm,config,*,policy='adaptive',normal='straight',particles=40,seed=901,use_gps=False,mounts=None,guard=None,raw_compare=None,motion_selector=None):
        if policy not in ('passive','fixed_straight','fixed_arc','adaptive') or normal not in CANDIDATES:
            raise ValueError('Geçersiz politika veya normal hareket.')
        self.mounts = dict(MOUNTS if mounts is None else mounts)
        self.raw_compare = raw_compare or compare_raw_hypotheses
        self.motion_selector = motion_selector or select_hypothesis_motion
        self.reference = make_backend(algorithm,particles,seed,self.mounts)
        self.fusion = make_backend(algorithm,particles,seed,self.mounts) if use_gps else None
        self.guard = guard if guard is not None else ScanAlignmentGuard(config)
        self.gps_monitor = GPSConsistencyMonitor()
        self.config,self.policy = config,policy
        self.normal_command = CANDIDATES[normal]
        self.time_s = 0.
        self.trigger_time_s = self.window_end_s = None
        self.selected_motion = self.decision = None
        self.trigger_source = None
        self.mode = 'NORMAL'
        self.last_time = None
        self.path,self.reference_path,self.history,self.raw_history = [],[],[],[]
        self.gps_rejected = 0

    def command(self):
        if self.trigger_time_s is None:
            return self.normal_command
        if self.time_s>=self.window_end_s-1e-9 or self.mode=='NO_SUPPORTED_MANEUVER':
            return (0.,0.)
        return CANDIDATES[self.selected_motion] if self.selected_motion else self.normal_command

    def process(self,events):
        if not events:
            raise ValueError('Boş ölçüm grubu.')
        time = float(events[0]['time_s'])
        if not math.isfinite(time) or time<0 or (self.last_time is not None and time<=self.last_time) or any(e['time_s']!=time or e['schema_version']!=1 for e in events):
            raise ValueError('Geçersiz ölçüm zamanı veya şema.')
        if len({(e['sensor_type'],e['sensor_id']) for e in events})!=len(events):
            raise ValueError('Tekrarlanan sensör.')
        backends = [self.reference]+([self.fusion] if self.fusion else [])
        encoder = next((e for e in events if e['sensor_type']=='encoder'),None)
        gyro = next((e for e in events if e['sensor_type']=='gyro'),None)
        if (encoder is None)!=(gyro is None):
            raise ValueError('Encoder/gyro çifti gerekli.')
        if encoder:
            for backend in backends:
                backend.predict(EncoderMeasurement(time_s=time,**encoder['data']),GyroMeasurement(time_s=time,**gyro['data']))
        view = geometry_view(self.reference)
        packets = sorted((e for e in events if e['sensor_type']=='lidar'),key=lambda e:e['sensor_id'])
        raw = None
        if packets:
            raw = self.raw_compare(view,packets,self.mounts,self.config)
            self.raw_history.append({'time_s':time,**raw})
            self.guard.observe(view,packets,self.mounts,time)
        for event in events:
            if event['sensor_type']=='gps':
                packet = GPSMeasurement(time_s=time,**event['data'])
                check = self.gps_monitor.observe(gps_reference(self.reference),packet)
                if self.fusion:
                    if packet.available and (check['instant_reject'] or self.gps_monitor.confirmed):
                        self.gps_rejected += 1
                    else:
                        self.fusion.update_gps(packet)
        if self.trigger_time_s is None and (self.guard.confirmed or self.gps_monitor.confirmed):
            self.trigger_time_s,self.window_end_s = time,time+2.
            self.trigger_source = 'geometry' if self.guard.confirmed else 'gps_reference'
            if self.policy=='adaptive':
                if self.guard.confirmed and raw is not None:
                    self.decision = self.motion_selector(view,raw,self.mounts)
                else:
                    self.decision = {'status':'no_supported_candidate','selected_motion':None,'reason':'gps_translation_hypothesis_not_implemented'}
                self.selected_motion = self.decision['selected_motion']
                self.mode = 'DIAGNOSING' if self.selected_motion else 'NO_SUPPORTED_MANEUVER'
            elif self.policy!='passive':
                self.selected_motion = self.policy.removeprefix('fixed_')
                self.mode = 'DIAGNOSING'
            else:
                self.mode = 'PASSIVE_OBSERVATION'
        if not self.guard.quarantine_active:
            for packet in packets:
                for backend in backends:
                    backend.update_scan(packet['sensor_id'],packet['data']['scan']['points'])
        if self.trigger_time_s is None:
            self.mode = 'WATCH' if self.guard.quarantine_active else 'NORMAL'
        elif time>=self.window_end_s-1e-9:
            self.mode = 'DIAGNOSIS_WINDOW_ENDED'
        self.time_s,self.last_time = time,time
        self.reference_path.append(self.reference.snapshot(time))
        self.path.append((self.fusion or self.reference).snapshot(time))
        self.history.append({'time_s':time,'mode':self.mode,'joint_status':joint_status(self.guard.confirmed,self.gps_monitor.confirmed),'selected_motion':self.selected_motion,'map_quarantined':self.guard.quarantine_active})
