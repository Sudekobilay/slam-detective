"""Tahmini haritada rakip hata modellerine göre hareket seçimi."""

import copy
from dataclasses import replace
import math
from types import SimpleNamespace

from estimation.diagnostic_motion import (
    CANDIDATES,
    _clearance,
)
from estimation.online_diagnostic_agent import (
    OnlineDiagnosticAgent,
)
from estimation.raw_scan_hypotheses import (
    compare_raw_hypotheses,
)
from simulator.encoder import EncoderMeasurement
from simulator.environment import Wall
from simulator.imu import GyroMeasurement
from simulator.lidar import Lidar, LidarConfig
from simulator.robot import DifferentialDriveRobot, Pose


def preview_state(estimator, events):
    preview = copy.deepcopy(estimator)

    encoder = next(
        (
            event
            for event in events
            if event["sensor_type"] == "encoder"
        ),
        None,
    )
    gyro = next(
        (
            event
            for event in events
            if event["sensor_type"] == "gyro"
        ),
        None,
    )

    if encoder is not None:
        preview.predict(
            EncoderMeasurement(
                time_s=encoder["time_s"],
                **encoder["data"],
            ),
            GyroMeasurement(
                time_s=gyro["time_s"],
                **gyro["data"],
            ),
        )

    return preview


def select_hypothesis_motion(estimator, raw, mounts):
    """En iyi iki modelin beklenen menzillerini karşılaştırır.

    Puan bilgi kazancı veya olasılık değildir.
    Açıklık kontrolü yalnız öğrenilmiş harita parçalarına bakar.
    """
    result = {
        "status": "no_supported_candidate",
        "selected_motion": None,
        "candidates": [],
        "models": [],
    }

    if raw["status"] != "evaluated":
        return result

    models = raw["models"][:2]
    result["models"] = copy.deepcopy(models)

    features = tuple(estimator.map_features())

    if not features:
        return result

    # Gerçek ortam değil, yalnız öğrenilmiş çizgiler.
    environment = SimpleNamespace(
        walls=tuple(
            Wall(feature.start, feature.end)
            for feature in features
        )
    )

    lidar = Lidar(
        LidarConfig(
            field_of_view=math.pi,
            beam_count=91,
            max_range=15.0,
        )
    )

    for name, rates in CANDIDATES.items():
        predictions = []
        safe = True

        for model in models:
            angle = math.radians(
                model["best_offset_deg"]
            )
            pose = estimator.pose

            if model["hypothesis"] == "heading_offset":
                pose = Pose(
                    pose.x,
                    pose.y,
                    pose.theta + angle,
                )

            robot = DifferentialDriveRobot(
                estimator.geometry,
                pose,
            )

            candidate_mounts = dict(mounts)

            for sensor in mounts:
                if (
                    model["hypothesis"]
                    == f"{sensor}_yaw"
                ):
                    candidate_mounts[sensor] = replace(
                        mounts[sensor],
                        yaw=mounts[sensor].yaw + angle,
                    )

            scans = []

            velocity = (
                estimator.geometry.left_radius * rates[0]
                + estimator.geometry.right_radius * rates[1]
            ) / 2.0

            required_clearance = (
                0.05 + abs(velocity) * 0.1
            )

            safe = safe and (
                _clearance(
                    robot.pose,
                    features,
                    0.35,
                ) > required_clearance
            )

            for step in range(1, 21):
                robot.step(*rates, 0.1)

                safe = safe and (
                    _clearance(
                        robot.pose,
                        features,
                        0.35,
                    ) > required_clearance
                )

                if step % 5 == 0:
                    scans.append({
                        sensor: lidar.scan(
                            mount.world_pose(robot.pose),
                            environment,
                        ).ranges
                        for sensor, mount
                        in candidate_mounts.items()
                    })

            predictions.append(scans)

        scores = []
        counts = []

        for first, second in zip(*predictions):
            differences = []
            frame_supported = True
            frame_counts = {}

            for sensor in ("front", "rear"):
                values = [
                    min(abs(a - b), 0.5) ** 2
                    for a, b in zip(
                        first[sensor],
                        second[sensor],
                    )
                    if a is not None and b is not None
                ]

                frame_counts[sensor] = len(values)

                frame_supported = (
                    frame_supported
                    and len(values) >= 20
                )

                differences.extend(values)

            counts.append(frame_counts)

            if frame_supported:
                scores.append(
                    sum(differences) / len(differences)
                )

        score = (
            sum(scores) / 4.0
            if len(scores) == 4
            else None
        )

        eligible = (
            safe
            and score is not None
            and score > 1e-12
        )

        result["candidates"].append({
            "motion": name,
            "eligible": bool(eligible),
            "known_map_clearance": bool(safe),
            "score_m2": (
                float(score)
                if score is not None
                else None
            ),
            "common_ray_counts": counts,
        })

    eligible = [
        row
        for row in result["candidates"]
        if row["eligible"]
    ]

    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                row["score_m2"],
                row["motion"] == "straight",
            ),
        )

        result.update(
            status="selected",
            selected_motion=chosen["motion"],
        )

    return result


class HypothesisMotionAgent(OnlineDiagnosticAgent):
    """Yeni politika yalnız bu alt sınıfta açılır."""

    def __init__(self, policy, normal_motion, config):
        from estimation.scan_alignment_guard import (
            ScanAlignmentGuard,
        )

        if policy not in (
            "passive",
            "fixed_straight",
            "fixed_arc",
            "adaptive",
        ):
            raise ValueError("Geçersiz politika.")

        super().__init__(
            strategy="passive",
            scan_guard=ScanAlignmentGuard(config),
        )

        self.policy = policy
        self.config = config
        self.normal_command = CANDIDATES[normal_motion]
        self.raw_history = []

    def process(self, events):
        packets = [
            event
            for event in events
            if event["sensor_type"] == "lidar"
        ]

        if packets:
            preview = preview_state(
                self.estimator,
                events,
            )

            raw = compare_raw_hypotheses(
                preview,
                packets,
                self.mounts,
                self.config,
            )

            self.raw_history.append({
                "time_s": events[0]["time_s"],
                **raw,
            })

        was_active = self.hold.active
        super().process(events)

        if not was_active and self.hold.active:
            if self.policy == "adaptive":
                self.decision = select_hypothesis_motion(
                    preview,
                    raw,
                    self.mounts,
                )
                self.selected_motion = (
                    self.decision["selected_motion"]
                )

            elif self.policy != "passive":
                self.selected_motion = (
                    self.policy.removeprefix("fixed_")
                )

            if (
                self.policy == "adaptive"
                and self.selected_motion is None
            ):
                self.mode = "NO_SUPPORTED_MANEUVER"

            elif self.selected_motion:
                self.mode = "DIAGNOSING"

            self.history[-1]["mode"] = self.mode

    def command(self):
        if (
            self.hold.active
            and self.estimator.time_s
            < self.window_end_s - 1e-9
        ):
            if self.mode == "NO_SUPPORTED_MANEUVER":
                return (0.0, 0.0)

            if self.selected_motion is not None:
                return CANDIDATES[self.selected_motion]

        return super().command()