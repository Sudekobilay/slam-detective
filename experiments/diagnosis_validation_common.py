"""Teşhis doğrulaması için kayıt üretimi ve sabit referans.

Gerçek ortam yalnız kayıt üretiminde kullanılır.
Yön ofseti kontrollü kestirim perturbasyonudur;
fiziksel gyro bias modeli değildir.
"""

import copy
import json
import math
from itertools import groupby

from estimation.hypothesis_motion import preview_state
from estimation.online_diagnostic_agent import (
    OnlineDiagnosticAgent,
)
from estimation.raw_scan_hypotheses import (
    compare_raw_hypotheses,
)
from experiments.exp_30_scan_guard_audit import (
    fixed_recording,
)
from simulator.environment import create_demo_environment
from simulator.lidar import LidarConfig
from simulator.measurement_stream import measurement_to_event
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit
from simulator.robot import Pose, wrap_angle


MOUNTS = {
    "front": SensorMount(x=0.25),
    "rear": SensorMount(x=-0.25, yaw=math.pi),
}


def fault_recording(normal, seed, source, angle):
    if source not in (
        "healthy",
        "front_yaw",
        "rear_yaw",
        "heading_offset",
    ):
        raise ValueError("Geçersiz hata kaynağı.")

    if (
        not math.isfinite(angle)
        or abs(angle) > 10
        or (source == "healthy" and angle != 0)
    ):
        raise ValueError("Geçersiz hata açısı.")

    events, records = fixed_recording(
        normal,
        seed,
        0.0,
    )

    if source not in ("front_yaw", "rear_yaw"):
        return events, records

    sensor = source.removesuffix("_yaw")
    mount = MOUNTS[sensor]

    unit = LidarUnit(
        sensor,
        SensorMount(
            mount.x,
            mount.y,
            mount.yaw + math.radians(angle),
        ),
        LidarConfig(
            field_of_view=math.pi,
            beam_count=91,
            max_range=15.0,
        ),
    )

    environment = create_demo_environment()

    truth = {
        round(row["time_s"], 12): row
        for row in records["truth"]
    }

    for index, event in enumerate(events):
        if (
            event["sensor_id"] == sensor
            and event["time_s"] >= 2.0
        ):
            row = truth[event["time_s"]]

            pose = Pose(
                row["x_m"],
                row["y_m"],
                row["theta_rad"],
            )

            packet = unit.measure(
                pose,
                environment,
                event["time_s"],
            )

            events[index] = json.loads(json.dumps(
                measurement_to_event(packet),
                allow_nan=False,
            ))

    return events, records


def learn_reference(events):
    agent = OnlineDiagnosticAgent(
        strategy="passive"
    )

    for time, batch in groupby(
        events,
        key=lambda event: event["time_s"],
    ):
        if time >= 2.0:
            break

        agent.process(list(batch))

    return copy.deepcopy(agent.estimator)


def inspect_frozen(
    events,
    reference,
    config,
    *,
    heading_perturbation_deg=0.0,
):
    """t>=2 harita sabit; hareket ön tahmini ilerler."""
    if not math.isfinite(heading_perturbation_deg):
        raise ValueError("Geçersiz yön perturbasyonu.")

    estimator = copy.deepcopy(reference)
    rows = []

    for time, batch in groupby(
        events,
        key=lambda event: event["time_s"],
    ):
        if time < 2.0:
            continue

        batch = list(batch)
        estimator = preview_state(estimator, batch)

        packets = [
            event
            for event in batch
            if event["sensor_type"] == "lidar"
        ]

        if packets:
            observed = copy.deepcopy(estimator)

            # Kontrollü test düzeneği yön tahminini bozar.
            # Hipotez işlevine gerçek poz veya hata
            # etiketi verilmez.
            observed.state[2] = wrap_angle(
                observed.state[2]
                + math.radians(
                    heading_perturbation_deg
                )
            )

            raw = compare_raw_hypotheses(
                observed,
                packets,
                MOUNTS,
                config,
            )

            rows.append({
                "time_s": time,
                "raw": raw,
            })

    return rows


def confirm_sequence(frames):
    """Aynı model/işarette iki ardışık aday gerekir.

    İlk doğrulama saklanır. Bu kural istatistiksel
    kesinlik anlamına gelmez.
    """
    previous = None
    streak = 0

    for row in frames:
        decision = row["decision"]

        if decision["status"] != "candidate":
            previous = None
            streak = 0
            continue

        key = (
            decision["hypothesis"],
            1 if decision["offset_deg"] > 0 else -1,
        )

        streak = (
            streak + 1
            if previous == key
            else 1
        )
        previous = key

        if streak >= 2:
            return {
                "status": "confirmed",
                "hypothesis": key[0],
                "confirmation_time_s": row["time_s"],
            }

    if frames and all(
        row["decision"]["status"]
        == "nominal_consistent"
        for row in frames
    ):
        return {
            "status": "nominal_consistent",
            "hypothesis": None,
            "confirmation_time_s": None,
        }

    return {
        "status": "uncertain",
        "hypothesis": None,
        "confirmation_time_s": None,
    }