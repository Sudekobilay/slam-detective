"""Compose agents and calibration without importing experiment entrypoints."""
from dataclasses import asdict
from functools import partial
from configuration.layouts import read_layout
from estimation.active_pipeline import ActivePipeline, make_backend, geometry_view
from estimation.configurable_geometry import LayoutScanGuard, compare_layout_hypotheses, select_layout_motion, usable_alignments
from estimation.diagnostic_motion import CANDIDATES
from estimation.scan_alignment_guard import ScanGuardConfig, scan_alignment
from simulator.encoder import EncoderMeasurement
from simulator.imu import GyroMeasurement
from simulator.layout_world import LayoutWorld

def create_agent(data, config, algorithm='ekf', **options):
    units = read_layout(data)
    mounts = {u.sensor_id: u.actual_mount for u in units}
    configs = {u.sensor_id: u.config for u in units}
    return ActivePipeline(
        algorithm, config, mounts=mounts, guard=LayoutScanGuard(config),
        raw_compare=compare_layout_hypotheses,
        motion_selector=partial(select_layout_motion, lidar_configs=configs), **options,
    )


def calibrate_layout(data, algorithm='ekf', particles=40):
    units = read_layout(data)
    mounts = {u.sensor_id: u.actual_mount for u in units}
    collection = ScanGuardConfig(19., 100.)
    gaps, rms = [], []
    for normal in CANDIDATES:
        for seed in range(11, 16):
            world = LayoutWorld(data, seed)
            backend = make_backend(algorithm, particles, 10000 + seed, mounts)
            batch = world.initial_measurements()
            for index in range(41):
                if index:
                    batch = world.step(*CANDIDATES[normal])
                    encoder = next(e for e in batch if e['sensor_type'] == 'encoder')
                    gyro = next(e for e in batch if e['sensor_type'] == 'gyro')
                    backend.predict(EncoderMeasurement(time_s=encoder['time_s'], **encoder['data']),
                                    GyroMeasurement(time_s=gyro['time_s'], **gyro['data']))
                packets = [e for e in batch if e['sensor_type'] == 'lidar']
                if packets:
                    view = geometry_view(backend)
                    alignments = {e['sensor_id']: scan_alignment(
                        view, e, mounts[e['sensor_id']], collection
                    ) for e in packets}
                    good = usable_alignments(alignments, collection)
                    rms.extend(row['alignment_rms_m'] for row in good.values())
                    if len(good) >= 2:
                        offsets = [row['offset_deg'] for row in good.values()]
                        gaps.append(max(offsets) - min(offsets))
                    for event in packets:
                        backend.update_scan(event['sensor_id'], event['data']['scan']['points'])
            print(f'Kalibrasyon: {data["name"]}/{algorithm}/{normal}/seed_{seed}', flush=True)
    if not rms or (len(units) >= 2 and not gaps):
        raise RuntimeError('Bu yerleşimde sağlıklı kalibrasyon desteği yetersiz.')
    config = ScanGuardConfig(max(gaps, default=0.) + .5, max(rms) + .01)
    return config, {'layout': data, 'algorithm': algorithm, 'particles': particles,
                    'healthy_sequences': 10, 'sensor_seeds': list(range(11, 16)),
                    'supported_pair_frames': len(gaps), 'config': asdict(config)}


def run_layout(data, config, *, algorithm='ekf', policy='adaptive', normal='straight',
               fault='lidar', sensor=None, yaw=5., seed=1001, particles=40):
    units = read_layout(data)
    sensor = units[0].sensor_id if sensor is None else sensor
    world = LayoutWorld(data, seed, fault, sensor, yaw)
    agent = create_agent(data, config, algorithm, policy=policy, normal=normal,
                         seed=seed + 20000, particles=particles, use_gps=data['gps_enabled'])
    from application.runner import run_episode
    return run_episode(world, agent, changed_commands=False).legacy()
