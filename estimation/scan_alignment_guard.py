"""İsteğe bağlı, ham tarama uyumundan sensörler arası açı tutarsızlığı izleme.

NIS kapısı değiştirilmez. Ek izleyici geometrik eşleştirmeden önce ham
noktaları öğrenilmiş sonlu çizgilere hizalar. Yeni referans gerçek harita
veya gerçek montaj bilgisi değildir. Bu bir deneysel alarm sinyalidir;
hata kaynağı teşhisi veya istatistiksel güven olasılığı değildir.
"""
from dataclasses import dataclass
import math

import numpy as np

from estimation.lidar_update_hold import LidarUpdateHold


@dataclass(frozen=True)
class ScanGuardConfig:
    relative_yaw_threshold_deg: float
    max_alignment_rms_m: float
    angle_limit_deg: float = 10.
    angle_step_deg: float = .5
    trim_fraction: float = .6
    min_points: int = 20
    consecutive_scans: int = 2

    def __post_init__(self):
        values=(self.relative_yaw_threshold_deg,self.max_alignment_rms_m,self.angle_limit_deg,self.angle_step_deg)
        if not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError('Geçersiz izleyici eşiği veya açı ızgarası.')
        if not self.angle_step_deg<=self.angle_limit_deg<=30.:
            raise ValueError('Açı aralığı 30° altında olmalı.')
        if self.relative_yaw_threshold_deg>=2*self.angle_limit_deg:
            raise ValueError('Göreli açı eşiği arama alanı dışında.')
        if not math.isfinite(self.trim_fraction) or not 0<self.trim_fraction<=1:
            raise ValueError('Geçersiz kırpma oranı.')
        if type(self.min_points) is not int or self.min_points<3:
            raise ValueError('En az üç nokta gerekli.')
        if type(self.consecutive_scans) is not int or self.consecutive_scans<2:
            raise ValueError('En az iki ardışık tarama gerekli.')
        if math.floor(self.angle_limit_deg/self.angle_step_deg)>200:
            raise ValueError('Açı arama ızgarası çok yoğun.')


def scan_alignment(estimator, packet, mount, config):
    """Durumu değiştirmeden sabit sayıda en yakın nokta uzaklığını puanlar."""
    features=estimator.map_features()
    points=np.asarray([point for point in packet['data']['scan']['points'] if point is not None],dtype=float)
    if not features or len(points)<config.min_points:
        return None
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('Geçersiz ham LiDAR noktaları.')
    starts=np.asarray([f.start for f in features],dtype=float)
    ends=np.asarray([f.end for f in features],dtype=float)
    directions=ends-starts
    squared=np.sum(directions**2,axis=1)
    valid=squared>1e-12
    starts,directions,squared=starts[valid],directions[valid],squared[valid]
    if not len(starts):
        return None
    n=int(math.floor(config.angle_limit_deg/config.angle_step_deg+1e-12))
    angles=np.asarray(sorted((i*config.angle_step_deg for i in range(-n,n+1)),key=lambda a:(abs(a),a)))
    sensor=mount.world_pose(estimator.pose)
    angle=sensor.theta+np.radians(angles)
    c,s=np.cos(angle)[:,None],np.sin(angle)[:,None]
    world=np.stack([
        c*points[None,:,0]-s*points[None,:,1]+sensor.x,
        s*points[None,:,0]+c*points[None,:,1]+sensor.y,
    ],axis=-1)
    offsets=world[:,:,None,:]-starts[None,None,:,:]
    fractions=np.clip(np.sum(offsets*directions,axis=-1)/squared,0.,1.)
    residual=offsets-fractions[:,:,:,None]*directions
    nearest=np.min(np.sum(residual**2,axis=-1),axis=2)
    keep=max(config.min_points,math.ceil(len(points)*config.trim_fraction))
    costs=np.mean(np.partition(nearest,keep-1,axis=1)[:,:keep],axis=1)
    best=int(np.argmin(costs))
    return {
        'offset_deg':float(angles[best]),
        'alignment_rms_m':float(math.sqrt(costs[best])),
        'nominal_rms_m':float(math.sqrt(costs[0])),
        'point_count':len(points),'retained_point_count':keep,
        'at_search_edge':math.isclose(abs(float(angles[best])),n*config.angle_step_deg),
    }


class ScanAlignmentGuard:
    def __init__(self, config):
        self.config=config
        self.quarantine_active=False
        self.confirmed=False
        self._streak=0
        self._previous_sign=None
        self._last_time=None
        self.last_result=None
        self.history=[]

    def observe(self, estimator, packets, mounts, time_s):
        if not math.isfinite(time_s) or time_s<0 or (self._last_time is not None and time_s<=self._last_time):
            raise ValueError('Geçersiz veya tekrarlanan tarama zamanı.')
        per_sensor={}
        for packet in packets:
            sensor=packet['sensor_id']
            if sensor not in mounts or sensor in per_sensor:
                raise ValueError('Tanımsız veya tekrarlanan LiDAR.')
            per_sensor[sensor]=scan_alignment(estimator,packet,mounts[sensor],self.config)
        return self.update_alignment_result(time_s,per_sensor)

    def update_alignment_result(self,time_s,per_sensor):
        """Ölçümden üretilmiş hizalama sonucuyla durum makinesini ilerletir."""
        if not math.isfinite(time_s) or time_s<0 or (self._last_time is not None and time_s<=self._last_time):
            raise ValueError('Geçersiz veya tekrarlanan tarama zamanı.')
        for alignment in per_sensor.values():
            if alignment is None:
                continue
            if (
                not math.isfinite(alignment['offset_deg'])
                or not math.isfinite(alignment['alignment_rms_m'])
                or alignment['alignment_rms_m'] < 0
                or type(alignment['at_search_edge']) is not bool
            ):
                raise ValueError('Geçersiz hizalama sonucu.')
        supported=all(per_sensor.get(sensor) is not None for sensor in ('front','rear'))
        if supported:
            supported=all(
                not per_sensor[s]['at_search_edge']
                and per_sensor[s]['alignment_rms_m']<=self.config.max_alignment_rms_m
                for s in ('front','rear')
            )
        gap=None
        if supported:
            gap=per_sensor['front']['offset_deg']-per_sensor['rear']['offset_deg']
            warning=abs(gap)>self.config.relative_yaw_threshold_deg
            if warning:
                sign=1 if gap>0 else -1
                self._streak=self._streak+1 if self._previous_sign==sign else 1
                self._previous_sign=sign
                self.quarantine_active=True
                if self._streak>=self.config.consecutive_scans:
                    self.confirmed=True
            else:
                self._streak=0
                self._previous_sign=None
                if not self.confirmed:
                    self.quarantine_active=False
            status='confirmed' if self.confirmed else ('watch' if warning else 'consistent')
        else:
            # Eksik destek, sağlıklı ölçüm olarak kabul edilmez. Varsa geçici
            # harita koruması sürer; ardışık doğrulama sayacı sıfırlanır.
            self._streak=0
            self._previous_sign=None
            status='insufficient_alignment_support'
        self._last_time=time_s
        result={
            'time_s':time_s,'status':status,'relative_yaw_deg':gap,
            'threshold_deg':self.config.relative_yaw_threshold_deg,
            'consecutive_scans':self._streak,
            'quarantine_active':self.quarantine_active,
            'confirmed':self.confirmed,'alignments':per_sensor,
        }
        self.last_result=result
        self.history.append(result)
        return result


class CombinedLidarGate:
    """Özgün NIS alarmı veya doğrulanmış tarama tutarsızlığında kilitlenir."""
    def __init__(self,nis_threshold,scan_guard):
        self._nis=LidarUpdateHold(nis_threshold)
        self.nis_threshold=self._nis.nis_threshold
        self.scan_guard=scan_guard
        self.active=False
        self.trigger_time_s=None
        self.trigger_evidence=[]
        self.trigger_source=None

    def observe(self,time_s,evidence):
        self._nis.observe(time_s,evidence)
        if self.active:
            return False
        if self._nis.active:
            self.trigger_source='matched_nis'
            self.trigger_evidence=[{'source':'matched_nis',**row} for row in self._nis.trigger_evidence]
        elif self.scan_guard.confirmed:
            self.trigger_source='scan_alignment_consensus'
            self.trigger_evidence=[{'source':self.trigger_source,'alignment_result':self.scan_guard.last_result}]
        else:
            return False
        self.active=True
        self.trigger_time_s=time_s
        return True
