"""CARMEN eski FLASER kayıtları: LiDAR ve kayda ekli odometri.

Işın açıları eski biçimde bulunmaz; profil açıkça belirtilir.
Odometri gerçek konum veya IMU/encoder ham ölçümü değildir.
"""
from dataclasses import dataclass
import gzip
import math
from pathlib import Path

from simulator.mounting import SensorMount
from simulator.robot import Pose, wrap_angle


@dataclass(frozen=True)
class CarmenScan:
    timestamp_s: float
    odometry: Pose
    laser_pose: Pose
    points: tuple
    line_number: int


def relative_pose(first, second):
    dx, dy = second.x - first.x, second.y - first.y
    c, s = math.cos(first.theta), math.sin(first.theta)
    return Pose(c * dx + s * dy, -s * dx + c * dy,
                wrap_angle(second.theta - first.theta))


def laser_mount(scan):
    offset = relative_pose(scan.odometry, scan.laser_pose)
    return SensorMount(offset.x, offset.y, offset.theta)


class CarmenLoader:
    def __init__(self, path, *, start_angle_deg=-90., angle_step_deg=.5,
                 max_range_m=80., expected_beams=360):
        self.path = Path(path)
        if not all(math.isfinite(v) for v in (start_angle_deg, angle_step_deg, max_range_m)):
            raise ValueError('Sonlu açı ve menzil gerekli.')
        if angle_step_deg <= 0 or max_range_m <= 0 or type(expected_beams) is not int or expected_beams < 2:
            raise ValueError('Geçersiz LiDAR profili.')
        self.start = math.radians(start_angle_deg)
        self.step = math.radians(angle_step_deg)
        self.max_range = max_range_m
        self.expected_beams = expected_beams
        self.counts = {}

    def __iter__(self):
        self.counts = {}
        previous = None
        opener = gzip.open if self.path.suffix.lower() == '.gz' else open
        with opener(self.path, 'rt', encoding='utf-8', errors='strict') as handle:
            for number, line in enumerate(handle, 1):
                tokens = line.split()
                if not tokens or tokens[0].startswith('#'):
                    continue
                kind = tokens[0]
                self.counts[kind] = self.counts.get(kind, 0) + 1
                if kind != 'FLASER':
                    continue
                try:
                    count = int(tokens[1])
                    if count != self.expected_beams or len(tokens) < count + 11:
                        raise ValueError('Işın sayısı/profil veya alan sayısı uyuşmuyor.')
                    ranges = [float(v) for v in tokens[2:2 + count]]
                    tail = tokens[2 + count:]
                    values = [float(v) for v in tail[:7]]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError('Sonlu olmayan poz veya zaman.')
                    laser, odometry = Pose(*values[:3]), Pose(*values[3:6])
                    timestamp = values[6]
                    if previous is not None and timestamp <= previous:
                        raise ValueError('FLASER zamanları kesin artmalı.')
                    previous = timestamp
                    points = tuple(
                        (r * math.cos(self.start + index * self.step),
                         r * math.sin(self.start + index * self.step))
                        if math.isfinite(r) and 0. < r < self.max_range else None
                        for index, r in enumerate(ranges)
                    )
                except (ValueError, IndexError) as error:
                    raise ValueError(f'Satır {number}: {error}') from error
                yield CarmenScan(timestamp, odometry, laser, points, number)
