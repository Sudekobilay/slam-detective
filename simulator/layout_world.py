"""Configurable sensor world; truth remains simulation-only."""
from dataclasses import replace
import json
import math
from configuration.layouts import read_layout
from simulator.measurement_stream import measurement_to_event
from simulator.multi_lidar import LidarUnit
from simulator.streaming_world import StreamingWorld, WorldConfig

class LayoutWorld(StreamingWorld):
    """Gerçek konum ve gerçek montaj yalnız simülasyon üretiminde kullanılır."""
    def __init__(self, data, seed, fault='healthy', sensor=None, yaw=5., *, environment=None, initial_pose=None, geometry=None):
        self.units = read_layout(data)
        ids = {unit.sensor_id for unit in self.units}
        if fault not in ('healthy', 'lidar', 'gps', 'combined'):
            raise ValueError('Geçersiz hata türü.')
        if fault in ('lidar', 'combined') and sensor not in ids:
            raise ValueError('Hatalı sensör yapılandırmada bulunamadı.')
        if fault in ('gps', 'combined') and not data['gps_enabled']:
            raise ValueError('GPS kapalıyken GPS hatası uygulanamaz.')
        if not math.isfinite(yaw) or abs(yaw) > 10:
            raise ValueError('Montaj hatası en çok 10 derece olmalı.')
        self.data, self.fault, self.fault_sensor, self.yaw = data, fault, sensor, yaw
        super().__init__(WorldConfig(gyro_seed=seed, gps_seed=seed + 1),
                         environment=environment, initial_pose=initial_pose, geometry=geometry)

    def _batch(self, motion_packets):
        base = super()._batch(motion_packets)
        events = [e for e in base if e['sensor_type'] != 'lidar'
                  and (self.data['gps_enabled'] or e['sensor_type'] != 'gps')]
        time = round(self._index * self.dt, 12)
        if self._index % 5 == 0:
            for nominal in self.units:
                unit = nominal
                if time >= 2. and self.fault in ('lidar', 'combined') and nominal.sensor_id == self.fault_sensor:
                    mount = replace(nominal.actual_mount,
                                    yaw=nominal.actual_mount.yaw + math.radians(self.yaw))
                    unit = LidarUnit(nominal.sensor_id, mount, nominal.config)
                events.append(measurement_to_event(unit.measure(self._robot.pose, self._environment, time)))
        if time >= 2. and self.fault in ('gps', 'combined'):
            for event in events:
                if event['sensor_type'] == 'gps' and event['data']['available']:
                    event['data']['x_m'] += .5
                    event['data']['y_m'] += .3
        events.sort(key=lambda e: (e['time_s'], e['sensor_type'], e['sensor_id']))
        return json.loads(json.dumps(events, allow_nan=False))

    def evaluation_records(self):
        records = super().evaluation_records()
        records['metadata']['layout'] = self.data
        records['metadata']['layout_fault'] = {
            'type': self.fault, 'sensor': self.fault_sensor,
            'yaw_deg': self.yaw if self.fault in ('lidar', 'combined') else 0.,
            'start_s': 2., 'gps_bias_m': [.5, .3] if self.fault in ('gps', 'combined') else [0., 0.],
        }
        return records
