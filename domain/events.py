"""Validate the entire synchronous batch before any estimator can mutate."""
from dataclasses import dataclass
import math

from domain.measurements import EncoderMeasurement
from domain.measurements import GPSMeasurement
from domain.measurements import GyroMeasurement


class MeasurementError(ValueError):
    """Malformed or incompatible measurement; safe to correct and retry."""


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class MeasurementBatch:
    time_s: float
    events: tuple
    encoder: EncoderMeasurement | None
    gyro: GyroMeasurement | None
    gps: tuple
    lidar: tuple


def decode_batch(events, mounts, *, last_time, reference_time):
    """Synchronous encoder+gyro contract; delayed/asynchronous fusion is unsupported."""
    if not isinstance(events, (list, tuple)) or not events:
        raise MeasurementError('Nonempty measurement batch required.')
    packets = {'encoder': None, 'gyro': None}
    gps, lidar, seen = [], [], set()
    time = None
    try:
        for event in events:
            if not isinstance(event, dict) or set(event) != {
                'schema_version', 'time_s', 'sensor_type', 'sensor_id', 'data'
            }:
                raise MeasurementError('Invalid event fields.')
            stamp = event['time_s']
            if type(event['schema_version']) is not int or event['schema_version'] != 1:
                raise MeasurementError('Unsupported event schema.')
            if not finite(stamp) or stamp < 0 or (time is not None and stamp != time):
                raise MeasurementError('All timestamps must be finite and equal.')
            time = stamp
            kind, identity, data = event['sensor_type'], event['sensor_id'], event['data']
            if kind not in ('encoder', 'gyro', 'gps', 'lidar'):
                raise MeasurementError(f'Unknown sensor type: {kind!r}')
            if not isinstance(identity, str) or not isinstance(data, dict):
                raise MeasurementError('Invalid sensor identity or data.')
            key = (kind, identity)
            if key in seen:
                raise MeasurementError('Duplicate sensor in batch.')
            seen.add(key)
            if kind != 'lidar' and identity != kind:
                raise MeasurementError('Unexpected motion/GPS sensor identity.')
            if kind == 'encoder':
                if not all(finite(data[k]) for k in ('dt_s',)):
                    raise MeasurementError('Invalid encoder duration.')
                packets[kind] = EncoderMeasurement(time_s=time, **data)
            elif kind == 'gyro':
                if not all(finite(data[k]) for k in ('dt_s', 'yaw_rate_rad_s', 'nominal_variance')):
                    raise MeasurementError('Invalid gyro values.')
                packets[kind] = GyroMeasurement(time_s=time, **data)
            elif kind == 'gps':
                if type(data['available']) is not bool or not finite(data['nominal_variance_m2']):
                    raise MeasurementError('Invalid GPS availability or variance.')
                if data['available'] and not all(finite(data[k]) for k in ('x_m', 'y_m')):
                    raise MeasurementError('Available GPS requires finite coordinates.')
                if not data['available'] and (data['x_m'] is not None or data['y_m'] is not None):
                    raise MeasurementError('Unavailable GPS must have null coordinates.')
                gps.append(GPSMeasurement(time_s=time, **data))
            else:
                if identity not in mounts or set(data) != {'scan'}:
                    raise MeasurementError('Unknown LiDAR or invalid scan fields.')
                scan = data['scan']
                if set(scan) != {'angles', 'ranges', 'points'}:
                    raise MeasurementError('Invalid LiDAR scan schema.')
                angles, ranges, points = (scan[k] for k in ('angles', 'ranges', 'points'))
                if not angles or len(angles) != len(ranges) or len(points) != len(ranges):
                    raise MeasurementError('Inconsistent LiDAR dimensions.')
                for angle, distance, point in zip(angles, ranges, points):
                    if not finite(angle):
                        raise MeasurementError('Nonfinite LiDAR angle.')
                    if distance is None:
                        if point is not None:
                            raise MeasurementError('Missing range must have a null point.')
                        continue
                    if not finite(distance) or distance < 0:
                        raise MeasurementError('Invalid LiDAR range.')
                    if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(finite(v) for v in point):
                        raise MeasurementError('Invalid LiDAR point.')
                    expected = (distance * math.cos(angle), distance * math.sin(angle))
                    if not all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8) for a, b in zip(point, expected)):
                        raise MeasurementError('LiDAR point and range disagree.')
                lidar.append(event)
        if last_time is not None and time <= last_time:
            raise MeasurementError('Batch time must increase.')
        encoder, gyro = packets['encoder'], packets['gyro']
        if (encoder is None) != (gyro is None):
            raise MeasurementError('Encoder/gyro pair required.')
        if encoder is not None:
            if not math.isclose(encoder.dt_s, gyro.dt_s, rel_tol=0., abs_tol=1e-9):
                raise MeasurementError('Encoder/gyro durations differ.')
            expected = reference_time + encoder.dt_s
        else:
            expected = reference_time
        if not math.isclose(time, expected, rel_tol=0., abs_tol=1e-9):
            raise MeasurementError('Batch and estimator clocks disagree.')
        return MeasurementBatch(float(time), tuple(events), encoder, gyro, tuple(gps), tuple(sorted(lidar, key=lambda e: e['sensor_id'])))
    except (KeyError, TypeError, OverflowError) as error:
        raise MeasurementError(f'Invalid measurement fields: {error}') from error
