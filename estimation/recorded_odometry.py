"""Kayda ekli SE(2) odometri farkını doğrudan çekirdeklere uygular.

Encoder sayımı veya IMU örneği uydurulmaz. Gürültü parametreleri bu
adaptörün nominal varsayımıdır; veri setinden kalibre edilmiş değildir.
"""
import math
import numpy as np

from estimation.fastslam1 import FastSLAM1Backend
from estimation.fastslam2 import FastSLAM2Backend
from simulator.robot import wrap_angle


def odometry_prediction(state, delta):
    result = state.copy()
    c, s = math.cos(state[2]), math.sin(state[2])
    result[0] += c * delta.x - s * delta.y
    result[1] += s * delta.x + c * delta.y
    result[2] = wrap_angle(state[2] + delta.theta)
    jacobian = np.eye(4)
    jacobian[0, 2] = -s * delta.x - c * delta.y
    jacobian[1, 2] = c * delta.x - s * delta.y
    rotation = np.eye(4)
    rotation[:2, :2] = [[c, -s], [s, c]]
    return result, jacobian, rotation


def predict_recorded_odometry(backend, delta, time_s, *, translation_density=.02,
                              rotation_density=.03):
    estimator = backend.estimator
    dt = time_s - estimator.time_s
    if not all(math.isfinite(v) for v in (delta.x, delta.y, delta.theta, time_s,
                                           translation_density, rotation_density)):
        raise ValueError('Sonlu odometri ve zaman gerekli.')
    if dt <= 0 or translation_density <= 0 or rotation_density <= 0:
        raise ValueError('Zaman artmalı; gürültü yoğunlukları pozitif olmalı.')
    local_noise = np.diag([translation_density**2 * dt] * 2
                          + [rotation_density**2 * dt, .001**2 * dt])
    if isinstance(backend, FastSLAM1Backend):
        if isinstance(backend, FastSLAM2Backend):
            backend.pending = []
        for particle in backend.particles:
            mean, _, rotation = odometry_prediction(particle.state, delta)
            prior = rotation @ local_noise @ rotation.T + np.eye(4) * 1e-12
            if isinstance(backend, FastSLAM2Backend):
                backend.pending.append((mean.copy(), prior))
            particle.state = backend.rng.multivariate_normal(mean, prior)
            particle.state[2] = wrap_angle(particle.state[2])
    else:
        mean, jacobian, rotation = odometry_prediction(estimator.state[:4], delta)
        covariance = estimator.P.copy()
        covariance[:4, :4] = jacobian @ estimator.P[:4, :4] @ jacobian.T
        covariance[:4, :4] += rotation @ local_noise @ rotation.T
        covariance[:4, 4:] = jacobian @ estimator.P[:4, 4:]
        covariance[4:, :4] = covariance[:4, 4:].T
        estimator.state[:4] = mean
        estimator.P = (covariance + covariance.T) / 2
    estimator.time_s = time_s
