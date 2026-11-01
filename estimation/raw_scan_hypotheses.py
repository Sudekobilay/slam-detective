"""Ham noktalardan geometrik hata modelleri.

Posterior olasılık veya kesin sensör teşhisi üretmez.
"""

import math

import numpy as np


def _cost_curve(
    points,
    mount,
    pose,
    starts,
    directions,
    lengths,
    angles,
    heading,
    trim,
):
    delta = np.radians(angles)

    theta = (
        pose.theta + delta
        if heading
        else np.full(len(delta), pose.theta)
    )

    origins = np.column_stack((
        pose.x
        + np.cos(theta) * mount.x
        - np.sin(theta) * mount.y,
        pose.y
        + np.sin(theta) * mount.x
        + np.cos(theta) * mount.y,
    ))

    yaw = theta + mount.yaw + (
        0.0 if heading else delta
    )

    cosine = np.cos(yaw)[:, None]
    sine = np.sin(yaw)[:, None]

    world = np.stack((
        cosine * points[:, 0] - sine * points[:, 1],
        sine * points[:, 0] + cosine * points[:, 1],
    ), axis=-1)

    world += origins[:, None, :]

    offset = world[:, :, None, :] - starts

    fraction = np.clip(
        np.sum(offset * directions, axis=-1) / lengths,
        0.0,
        1.0,
    )

    residual = (
        offset - fraction[..., None] * directions
    )

    nearest = np.min(
        np.sum(residual ** 2, axis=-1),
        axis=2,
    )

    keep = max(20, math.ceil(len(points) * trim))

    return np.mean(
        np.partition(
            nearest,
            keep - 1,
            axis=1,
        )[:, :keep],
        axis=1,
    )


def compare_raw_hypotheses(
    estimator,
    packets,
    mounts,
    config,
):
    """Üç tek-hata modeli aynı noktalar ve haritada karşılaştırılır."""
    result = {
        "status": "insufficient_support",
        "nominal_cost_m2": None,
        "models": [],
    }

    features = tuple(estimator.map_features())

    if not features:
        return result

    selected = {}

    for packet in packets:
        sensor = packet["sensor_id"]

        if (
            sensor in selected
            or sensor not in ("front", "rear")
        ):
            raise ValueError(
                "Tekil front/rear taramaları gerekli."
            )

        raw = [
            point
            for point in packet["data"]["scan"]["points"]
            if point is not None
        ]
        points = np.asarray(raw, dtype=float)

        if len(points) < max(20, config.min_points):
            return result

        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            raise ValueError("Geçersiz ham noktalar.")

        selected[sensor] = points

    if set(selected) != {"front", "rear"}:
        return result

    starts = np.asarray(
        [feature.start for feature in features],
        dtype=float,
    )
    directions = np.asarray(
        [feature.end for feature in features],
        dtype=float,
    ) - starts

    lengths = np.sum(directions ** 2, axis=1)
    valid = lengths > 1e-12

    starts = starts[valid]
    directions = directions[valid]
    lengths = lengths[valid]

    if not len(starts):
        return result

    count = math.floor(
        config.angle_limit_deg
        / config.angle_step_deg
        + 1e-12
    )

    angles = np.asarray(sorted(
        (
            index * config.angle_step_deg
            for index in range(-count, count + 1)
        ),
        key=lambda value: (abs(value), value),
    ))

    curves = {}
    common = {}

    for sensor, points in selected.items():
        curves[sensor] = _cost_curve(
            points,
            mounts[sensor],
            estimator.pose,
            starts,
            directions,
            lengths,
            angles,
            False,
            config.trim_fraction,
        )

        common[sensor] = _cost_curve(
            points,
            mounts[sensor],
            estimator.pose,
            starts,
            directions,
            lengths,
            angles,
            True,
            config.trim_fraction,
        )

    nominal = (
        curves["front"][0] + curves["rear"][0]
    ) / 2.0

    result.update(
        status="evaluated",
        nominal_cost_m2=float(nominal),
    )

    for model in (
        "front_yaw",
        "rear_yaw",
        "heading_offset",
    ):
        if model == "heading_offset":
            front = common["front"]
            rear = common["rear"]
        else:
            front = (
                curves["front"]
                if model == "front_yaw"
                else np.full(
                    len(angles),
                    curves["front"][0],
                )
            )
            rear = (
                curves["rear"]
                if model == "rear_yaw"
                else np.full(
                    len(angles),
                    curves["rear"][0],
                )
            )

        costs = (front + rear) / 2.0
        best = int(np.argmin(costs))

        result["models"].append({
            "hypothesis": model,
            "best_offset_deg": float(angles[best]),
            "cost_m2": float(costs[best]),
            "improvement_m2": float(
                nominal - costs[best]
            ),
            "max_sensor_rms_m": float(
                math.sqrt(
                    max(front[best], rear[best])
                )
            ),
            "at_search_edge": bool(
                math.isclose(
                    abs(angles[best]),
                    max(abs(angles)),
                )
            ),
        })

    result["models"].sort(
        key=lambda row: row["cost_m2"]
    )

    return result


def diagnose_raw(result, thresholds, max_rms_m):
    """Tek karelik aday kararı; kesin teşhis değildir."""
    decision = {
        "status": "uncertain",
        "hypothesis": None,
        "offset_deg": None,
    }

    if result["status"] != "evaluated":
        return {
            **decision,
            "reason": "insufficient_support",
        }

    best, runner = result["models"][:2]

    if (
        best["improvement_m2"]
        <= thresholds["minimum_improvement_m2"]
    ):
        if (
            math.sqrt(result["nominal_cost_m2"])
            <= max_rms_m
        ):
            return {
                **decision,
                "status": "nominal_consistent",
                "reason": "no_supported_improvement",
            }

        return {
            **decision,
            "reason": "poor_nominal_fit",
        }

    if (
        best["at_search_edge"]
        or best["max_sensor_rms_m"] > max_rms_m
    ):
        return {
            **decision,
            "reason": "poor_or_boundary_fit",
        }

    if (
        runner["cost_m2"] - best["cost_m2"]
        <= thresholds["minimum_model_gap_m2"]
    ):
        return {
            **decision,
            "reason": "competing_models_close",
        }

    return {
        "status": "candidate",
        "hypothesis": best["hypothesis"],
        "offset_deg": best["best_offset_deg"],
        "reason": "geometric_support",
    }