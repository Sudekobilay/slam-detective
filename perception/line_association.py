"""Doğru özelliklerini dönüştürür ve sabit referansla eşleştirir."""

from dataclasses import dataclass, replace
import math

from perception.line_features import LineFeature
from simulator.mounting import transform_point
from simulator.robot import Pose


@dataclass(frozen=True)
class AssociationConfig:
    max_angle_rad: float = math.radians(5.0)
    max_perpendicular_m: float = 0.10
    max_gap_m: float = 0.30
    ambiguity_margin: float = 0.25

    def __post_init__(self):
        values = (
            self.max_angle_rad,
            self.max_perpendicular_m,
            self.max_gap_m,
        )

        if not all(
            math.isfinite(value) and value > 0.0
            for value in values
        ):
            raise ValueError("Eşleştirme eşikleri pozitif olmalıdır.")

        if self.max_angle_rad >= math.pi / 2:
            raise ValueError("Açı eşiği 90 dereceden küçük olmalıdır.")

        if not (
            math.isfinite(self.ambiguity_margin)
            and self.ambiguity_margin >= 0.0
        ):
            raise ValueError("Geçersiz belirsizlik payı.")


def transform_line(line: LineFeature, frame_pose: Pose):
    """Yerel doğruyu üst koordinat sistemine taşır."""
    cosine = math.cos(frame_pose.theta)
    sine = math.sin(frame_pose.theta)
    nx, ny = line.normal

    normal = (
        cosine * nx - sine * ny,
        sine * nx + cosine * ny,
    )

    # n_world · p_world = rho_local + n_world · translation
    rho = (
        line.rho_m
        + normal[0] * frame_pose.x
        + normal[1] * frame_pose.y
    )

    return replace(
        line,
        normal=normal,
        rho_m=rho,
        start=transform_point(line.start, frame_pose),
        end=transform_point(line.end, frame_pose),
    )


def candidate_score(observation, reference, config):
    """Uygun olmayan çift için None, diğerleri için geometrik puan."""
    on = observation.normal
    rn = reference.normal

    dot = on[0] * rn[0] + on[1] * rn[1]
    cross = on[0] * rn[1] - on[1] * rn[0]

    # (n, rho) ve (-n, -rho) aynı doğruyu temsil eder.
    angle = math.atan2(abs(cross), abs(dot))

    observation_center = (
        (observation.start[0] + observation.end[0]) / 2,
        (observation.start[1] + observation.end[1]) / 2,
    )
    reference_center = (
        (reference.start[0] + reference.end[0]) / 2,
        (reference.start[1] + reference.end[1]) / 2,
    )

    perpendicular = 0.5 * (
        abs(
            rn[0] * observation_center[0]
            + rn[1] * observation_center[1]
            - reference.rho_m
        )
        + abs(
            on[0] * reference_center[0]
            + on[1] * reference_center[1]
            - observation.rho_m
        )
    )

    # Aynı sonsuz doğru üzerindeki uzak parçaları ayır.
    tangent = (-rn[1], rn[0])

    def projected_interval(line):
        values = [
            point[0] * tangent[0] + point[1] * tangent[1]
            for point in (line.start, line.end)
        ]
        return min(values), max(values)

    observation_interval = projected_interval(observation)
    reference_interval = projected_interval(reference)

    gap = max(
        0.0,
        max(observation_interval[0], reference_interval[0])
        - min(observation_interval[1], reference_interval[1]),
    )

    if (
        angle > config.max_angle_rad
        or perpendicular > config.max_perpendicular_m
        or gap > config.max_gap_m
    ):
        return None

    return (
        (angle / config.max_angle_rad)**2
        + (perpendicular / config.max_perpendicular_m)**2
        + (gap / config.max_gap_m)**2
    )


def associate_lines(observations, references, config=None):
    """
    Her gözleme durum ve varsa referans indeksi verir.

    Bir taramada aynı referans en fazla bir gözlemle eşleşir.
    Basit açgözlü atamadır; küresel optimum garanti edilmez.
    """
    config = config if config is not None else AssociationConfig()
    results = []
    proposals = []

    for observation_index, observation in enumerate(observations):
        candidates = []

        for reference_index, reference in enumerate(references):
            score = candidate_score(observation, reference, config)

            if score is not None:
                candidates.append((score, reference_index))

        candidates.sort()

        record = {
            "observation_index": observation_index,
            "reference_index": None,
            "status": "no_candidate",
            "score": None,
        }

        if candidates:
            record["score"] = candidates[0][0]

            ambiguous = (
                len(candidates) > 1
                and candidates[1][0] - candidates[0][0]
                <= config.ambiguity_margin
            )

            if ambiguous:
                record["status"] = "ambiguous"
            else:
                record["status"] = "conflict"
                proposals.append((
                    candidates[0][0],
                    observation_index,
                    candidates[0][1],
                ))

        results.append(record)

    used_references = set()

    for score, observation_index, reference_index in sorted(proposals):
        if reference_index in used_references:
            continue

        used_references.add(reference_index)
        results[observation_index].update(
            reference_index=reference_index,
            status="matched",
        )

    return results