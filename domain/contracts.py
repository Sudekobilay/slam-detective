"""Small runtime contracts; no simulator truth or experiment dependencies."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class PipelineMode(StrEnum):
    NORMAL = 'NORMAL'
    WATCH = 'WATCH'
    DIAGNOSING = 'DIAGNOSING'
    PASSIVE_OBSERVATION = 'PASSIVE_OBSERVATION'
    NO_SUPPORTED_MANEUVER = 'NO_SUPPORTED_MANEUVER'
    DIAGNOSIS_WINDOW_ENDED = 'DIAGNOSIS_WINDOW_ENDED'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class GeometrySnapshot:
    pose: object
    geometry: object
    features: tuple

    def map_features(self):
        return list(self.features)


@dataclass(frozen=True)
class ReferenceDistribution:
    state: object
    P: object
    time_s: float


@dataclass(frozen=True)
class BackendCapabilities:
    mapping: bool = True
    gps_update: bool = True
    geometry_snapshot: bool = True
    # Available interface does not certify fault-detection performance.
    diagnostic_performance_certified: bool = False


@dataclass(frozen=True)
class MotionDecision:
    status: str
    selected_motion: str | None
    evidence: dict

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get('status'), str):
            raise ValueError('Motion decision must declare status.')
        motion = value.get('selected_motion')
        if motion not in (None, 'straight', 'arc'):
            raise ValueError('Unsupported motion decision.')
        return cls(value['status'], motion, dict(value))


@runtime_checkable
class SlamBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    @property
    def clock_s(self) -> float: ...
    def predict(self, encoder, gyro) -> None: ...
    def update_gps(self, packet): ...
    def update_scan(self, sensor_id, points): ...
    def snapshot(self, time_s) -> dict: ...
    def geometry_snapshot(self) -> GeometrySnapshot: ...
    def reference_distribution(self) -> ReferenceDistribution: ...
