"""NumPy ile 2B noktadan noktaya ICP tarama eşleştirme."""

import math
import numpy as np


def icp_2d(
    source,
    target,
    initial_translation=(0.0, 0.0),
    initial_yaw=0.0,
    max_distance=0.5,
    trim_fraction=0.85,
    min_pairs=6,
    max_iterations=60,
    tolerance=1e-7,
):
    """
    source noktalarını target koordinatına taşır.

    Dönüşüm:
        target_point ≈ R @ source_point + translation

    Yakın eşleşmelerin bir kısmı tutulur.
    Bu basit sürüm tekil eşleşme veya kovaryans üretmez.
    """
    source = np.array(source, dtype=float, copy=True)
    target = np.array(target, dtype=float, copy=True)

    for points in (source, target):
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            raise ValueError("Noktalar sonlu N×2 dizileri olmalıdır.")

    translation = np.array(
        initial_translation, dtype=float, copy=True
    )

    if (
        translation.shape != (2,)
        or not np.isfinite(translation).all()
        or not math.isfinite(initial_yaw)
    ):
        raise ValueError("Geçersiz başlangıç dönüşümü.")

    if not (
        math.isfinite(max_distance)
        and max_distance > 0.0
        and math.isfinite(trim_fraction)
        and 0.0 < trim_fraction <= 1.0
        and type(min_pairs) is int
        and min_pairs >= 3
        and type(max_iterations) is int
        and max_iterations > 0
        and math.isfinite(tolerance)
        and tolerance > 0.0
    ):
        raise ValueError("Geçersiz ICP parametreleri.")

    cosine = math.cos(initial_yaw)
    sine = math.sin(initial_yaw)
    rotation = np.array([
        [cosine, -sine],
        [sine, cosine],
    ])

    def correspondences():
        transformed = source @ rotation.T + translation

        squared_distances = np.sum(
            (transformed[:, None, :] - target[None, :, :])**2,
            axis=2,
        )
        nearest = np.argmin(squared_distances, axis=1)
        squared_errors = squared_distances[
            np.arange(len(source)), nearest
        ]

        candidates = np.flatnonzero(
            squared_errors <= max_distance**2
        )

        if len(candidates) < min_pairs:
            return transformed, np.array([], dtype=int), nearest

        order = candidates[np.argsort(squared_errors[candidates])]
        keep = max(
            min_pairs,
            int(math.floor(trim_fraction * len(order))),
        )

        return transformed, order[:keep], nearest

    def result(converged, reason, iterations):
        if len(source) < min_pairs or len(target) < min_pairs:
            count = 0
            rmse = None
        else:
            transformed, selected, nearest = correspondences()
            count = len(selected)
            rmse = (
                float(np.sqrt(np.mean(np.sum(
                    (
                        transformed[selected]
                        - target[nearest[selected]]
                    )**2,
                    axis=1,
                ))))
                if count else None
            )

        return {
            "converged": converged,
            "reason": reason,
            "iterations": iterations,
            "matched_points": count,
            "source_point_count": len(source),
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "rmse_m": rmse,
        }

    if len(source) < min_pairs or len(target) < min_pairs:
        return result(False, "not_enough_points", 0)

    for iteration in range(1, max_iterations + 1):
        transformed, selected, nearest = correspondences()

        if len(selected) < min_pairs:
            return result(False, "not_enough_matches", iteration)

        matched_source = transformed[selected]
        matched_target = target[nearest[selected]]

        source_center = matched_source.mean(axis=0)
        target_center = matched_target.mean(axis=0)

        covariance = (
            (matched_source - source_center).T
            @ (matched_target - target_center)
        )

        U, _, Vt = np.linalg.svd(covariance)
        delta_rotation = Vt.T @ U.T

        # Yansıma yerine gerçek bir dönme matrisi kullan.
        if np.linalg.det(delta_rotation) < 0.0:
            Vt[-1, :] *= -1.0
            delta_rotation = Vt.T @ U.T

        delta_translation = (
            target_center - delta_rotation @ source_center
        )

        rotation = delta_rotation @ rotation
        translation = (
            delta_rotation @ translation + delta_translation
        )

        angle_change = abs(math.atan2(
            delta_rotation[1, 0],
            delta_rotation[0, 0],
        ))

        if (
            np.linalg.norm(delta_translation) < tolerance
            and angle_change < tolerance
        ):
            return result(True, "small_increment", iteration)

    return result(False, "iteration_limit", max_iterations)