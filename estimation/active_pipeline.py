"""Üç SLAM çekirdeği için genişletilebilir çevrim içi denetleyici.

Kararlar GPS almayan bağımsız haritalamadan gelir. İsteğe bağlı ikinci
kestirici korumalı GPS füzyonu yapar; denetim referansını değiştirmez.
"""
from domain.events import decode_batch
from domain.contracts import PipelineMode, MotionDecision
from storage.recorders import MemoryRecorder

from estimation.diagnostic_motion import CANDIDATES
from estimation.gps_consistency import GPSConsistencyMonitor,joint_status
from estimation.hypothesis_motion import select_hypothesis_motion
from estimation.raw_scan_hypotheses import compare_raw_hypotheses
from estimation.scan_alignment_guard import ScanAlignmentGuard
from simulator.gps import GPSMeasurement


from estimation.backends import MOUNTS, make_backend, geometry_view, gps_reference


class ActivePipeline:
    def __init__(self,algorithm,config,*,policy='adaptive',normal='straight',particles=40,seed=901,use_gps=False,mounts=None,guard=None,raw_compare=None,motion_selector=None,initial_pose=None,initial_covariance=None,geometry=None,recorder=None):
        if policy not in ('passive','fixed_straight','fixed_arc','adaptive') or normal not in CANDIDATES:
            raise ValueError('Geçersiz politika veya normal hareket.')
        self.mounts = dict(MOUNTS if mounts is None else mounts)
        self.raw_compare = raw_compare or compare_raw_hypotheses
        self.motion_selector = motion_selector or select_hypothesis_motion
        self.reference = make_backend(algorithm,particles,seed,self.mounts, initial_pose=initial_pose, initial_covariance=initial_covariance, geometry=geometry)
        self.fusion = make_backend(algorithm,particles,seed,self.mounts, initial_pose=initial_pose, initial_covariance=initial_covariance, geometry=geometry) if use_gps else None
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
        self.recorder = recorder if recorder is not None else MemoryRecorder()
        self.path = self.recorder.stream('path')
        self.reference_path = self.recorder.stream('reference_path')
        self.history = self.recorder.stream('history')
        self.raw_history = self.recorder.stream('raw_history')
        self.failed_error = None
        self.gps_rejected = 0

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        value = PipelineMode(value)
        previous = getattr(self, '_mode', None)
        allowed = {
            PipelineMode.NORMAL: {PipelineMode.NORMAL, PipelineMode.WATCH, PipelineMode.DIAGNOSING,
                                 PipelineMode.PASSIVE_OBSERVATION, PipelineMode.NO_SUPPORTED_MANEUVER},
            PipelineMode.WATCH: {PipelineMode.NORMAL, PipelineMode.WATCH, PipelineMode.DIAGNOSING,
                                PipelineMode.PASSIVE_OBSERVATION, PipelineMode.NO_SUPPORTED_MANEUVER},
            PipelineMode.DIAGNOSING: {PipelineMode.DIAGNOSING, PipelineMode.DIAGNOSIS_WINDOW_ENDED},
            PipelineMode.PASSIVE_OBSERVATION: {PipelineMode.PASSIVE_OBSERVATION, PipelineMode.DIAGNOSIS_WINDOW_ENDED},
            PipelineMode.NO_SUPPORTED_MANEUVER: {PipelineMode.NO_SUPPORTED_MANEUVER, PipelineMode.DIAGNOSIS_WINDOW_ENDED},
            PipelineMode.DIAGNOSIS_WINDOW_ENDED: {PipelineMode.DIAGNOSIS_WINDOW_ENDED},
            PipelineMode.FAILED: {PipelineMode.FAILED},
        }
        if previous is not None and value != PipelineMode.FAILED and value not in allowed[previous]:
            raise RuntimeError(f'Invalid pipeline transition: {previous} -> {value}')
        self._mode = value

    def command(self):
        if self.failed_error is not None:
            return (0., 0.)
        if self.trigger_time_s is None:
            return self.normal_command
        if self.time_s>=self.window_end_s-1e-9 or self.mode=='NO_SUPPORTED_MANEUVER':
            return (0.,0.)
        return CANDIDATES[self.selected_motion] if self.selected_motion else self.normal_command

    def process(self, events):
        if self.failed_error is not None:
            raise RuntimeError('Pipeline failed; create a fresh instance.')
        batch = decode_batch(events, self.mounts, last_time=self.last_time,
                             reference_time=self.reference.clock_s)
        try:
            self._apply_batch(batch)
        except Exception as error:
            # Input errors are caught before this boundary. Unexpected numerical
            # or callback failures poison the instance; partial state is not retried.
            self.failed_error = f'{type(error).__name__}: {error}'
            self.mode = 'FAILED'
            raise

    def _apply_batch(self, batch):
        events, time = batch.events, batch.time_s
        backends = [self.reference] + ([self.fusion] if self.fusion else [])
        if batch.encoder is not None:
            for backend in backends:
                backend.predict(batch.encoder, batch.gyro)
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
                self.decision_record = MotionDecision.from_mapping(self.decision)
                self.selected_motion = self.decision_record.selected_motion
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
