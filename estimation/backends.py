"""Composition boundary for concrete SLAM implementations."""
import math
from estimation.fastslam1 import FastSLAM1Backend
from estimation.fastslam2 import FastSLAM2Backend
from estimation.slam_backend import LineSlamBackend
from simulator.mounting import SensorMount

MOUNTS = {'front': SensorMount(x=.25), 'rear': SensorMount(x=-.25, yaw=math.pi)}


def make_backend(algorithm, particles=40, seed=901, mounts=None, *,
                 initial_pose=None, initial_covariance=None, geometry=None):
    mounts = MOUNTS if mounts is None else mounts
    options = dict(initial_pose=initial_pose, initial_covariance=initial_covariance, geometry=geometry)
    if algorithm == 'ekf':
        return LineSlamBackend(mounts, **options)
    classes = {'fastslam1': FastSLAM1Backend, 'fastslam2': FastSLAM2Backend}
    if algorithm not in classes:
        raise ValueError('Unknown SLAM algorithm.')
    return classes[algorithm](mounts, particle_count=particles, seed=seed, **options)


def geometry_view(backend):
    return backend.geometry_snapshot()


def gps_reference(backend):
    return backend.reference_distribution()
