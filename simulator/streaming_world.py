"""Komut geldikçe ilerleyen 2B sensör dünyası.

Gerçek konum, hata ayarı ve kusursuz ortam bu modülde kalır.
Denetleyiciye yalnız returned measurement batches verilir.
"""
from dataclasses import asdict, dataclass
import json
import math

from simulator.collision import try_step
from simulator.encoder import EncoderConfig, WheelEncoder
from simulator.environment import create_demo_environment
from simulator.gps import GPSConfig, GPSSensor
from simulator.imu import GyroConfig, Gyroscope
from simulator.lidar import LidarConfig
from simulator.measurement_stream import measurement_to_event
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit, scan_all
from simulator.robot import DifferentialDriveRobot, Pose


@dataclass(frozen=True)
class WorldConfig:
    front_yaw_error_deg: float = 0.0
    fault_start_s: float = 2.0
    gyro_seed: int = 101
    gps_seed: int = 102

    def __post_init__(self):
        if (
            not math.isfinite(self.front_yaw_error_deg)
            or abs(self.front_yaw_error_deg) > 10
            or not math.isfinite(self.fault_start_s)
            or self.fault_start_s < 0
        ):
            raise ValueError('Geçersiz dünya yapılandırması.')


class StreamingWorld:
    dt = 0.1

    def __init__(self, config=None, *, environment=None, initial_pose=None, geometry=None):
        self._config = config or WorldConfig()
        self._environment = environment if environment is not None else create_demo_environment()
        self._robot = DifferentialDriveRobot(initial_pose=initial_pose if initial_pose is not None else Pose(1., 3., 0.), geometry=geometry)
        self._encoder = WheelEncoder(EncoderConfig())
        self._gyro = Gyroscope(GyroConfig(noise_std_rad_s=.01), seed=self._config.gyro_seed)
        self._gps = GPSSensor(GPSConfig(noise_std_m=.05, outage_start_s=1., outage_end_s=3.), seed=self._config.gps_seed)
        lidar = LidarConfig(field_of_view=math.pi, beam_count=91, max_range=15.)
        self._nominal_units = (
            LidarUnit('front', SensorMount(x=.25), lidar),
            LidarUnit('rear', SensorMount(x=-.25, yaw=math.pi), lidar),
        )
        self._fault_units = (
            LidarUnit('front', SensorMount(x=.25, yaw=math.radians(self._config.front_yaw_error_deg)), lidar),
            self._nominal_units[1],
        )
        self._index = 0
        self._initialized = False
        self._truth = []
        self._controls = []
        self._affected_scans = []

    def _batch(self, motion_packets):
        time_s = self._index * self.dt
        events = [measurement_to_event(packet) for packet in motion_packets]
        self._truth.append({
            'time_s': time_s, 'x_m': self._robot.pose.x,
            'y_m': self._robot.pose.y, 'theta_rad': self._robot.pose.theta,
        })
        if self._index % 2 == 0:
            events.append(measurement_to_event(self._gps.measure(self._robot.pose, time_s)))
        if self._index % 5 == 0:
            active = self._config.front_yaw_error_deg != 0. and time_s >= self._config.fault_start_s
            if active:
                self._affected_scans.append(time_s)
            for packet in scan_all(self._fault_units if active else self._nominal_units, self._robot.pose, self._environment, time_s):
                events.append(measurement_to_event(packet))
        events.sort(key=lambda event: (event['time_s'], event['sensor_type'], event['sensor_id']))
        return json.loads(json.dumps(events, allow_nan=False))

    def initial_measurements(self):
        if self._initialized:
            raise ValueError('Başlangıç ölçümleri yalnız bir kez alınabilir.')
        self._initialized = True
        return self._batch([])

    def step(self, left_rate, right_rate):
        if not self._initialized:
            raise ValueError('Önce başlangıç ölçümleri alınmalı.')
        if not math.isfinite(left_rate) or not math.isfinite(right_rate):
            raise ValueError('Tekerlek komutları sonlu olmalı.')
        if not try_step(self._robot, self._environment, left_rate, right_rate, self.dt, radius=.35):
            raise RuntimeError('Gerçek simülatörün çarpışma kontrolü komutu reddetti.')
        geometry = self._robot.geometry
        yaw_rate = (geometry.right_radius * right_rate - geometry.left_radius * left_rate) / geometry.wheelbase
        self._index += 1
        self._controls.append({
            'interval_end_s': self._index * self.dt,
            'left_rate_rad_s': left_rate, 'right_rate_rad_s': right_rate,
        })
        return self._batch([
            self._encoder.measure(left_rate, right_rate, self.dt),
            self._gyro.measure(yaw_rate, self.dt),
        ])

    def evaluation_records(self):
        """Yalnız deney yöneticisinin çevrimdışı değerlendirmesi içindir."""
        return json.loads(json.dumps({
            'truth': self._truth, 'controls': self._controls,
            'metadata': {
                'world_config': asdict(self._config),
                'dt_s': self.dt,
                'robot_geometry': asdict(self._robot.geometry),
                'affected_front_scan_times_s': self._affected_scans,
                'environment': {
                    'boundary': asdict(self._environment.boundary),
                    'obstacles': [asdict(o) for o in self._environment.obstacles],
                },
            },
        }, allow_nan=False))
