"""Sabit eşleşmeler ve ortak ön tahmin üzerinde geometrik hipotez karşılaştırması.

Bu modül montajı kalibre etmez, filtreyi değiştirmez ve kesin teşhis üretmez.
GPS sapması bu hipotez kümesinde henüz modellenmez.
"""

from dataclasses import replace
import math

import numpy as np

from estimation.line_ekf_slam import line_observation
from simulator.robot import wrap_angle


def compare_mounting_hypotheses(
    estimator, observations, *, angle_limit_deg=10.0, angle_step_deg=0.5
):
    """observations: (sensor_id, landmark_id, feature, nominal_mount).

    Bütün adaylar aynı, nominal montajla seçilmiş eşleşmeleri kullanır.
    Ortak kovaryans, farklı çizgiler arasındaki çapraz terimleri de içerir.
    NLL yalnız bu sabit eşleşmeler ve nominal Gaussian model koşulundadır;
    posterior hata olasılığı veya doğrulanmış karar eşiği değildir.
    """
    if (
        not math.isfinite(angle_limit_deg)
        or not math.isfinite(angle_step_deg)
        or angle_limit_deg <= 0.0
        or angle_limit_deg > 45.0
        or angle_step_deg <= 0.0
        or angle_step_deg > angle_limit_deg
    ):
        raise ValueError("Geçersiz açı arama sınırı veya adımı.")

    observations = tuple(observations)
    counts = {sensor: 0 for sensor in ("front", "rear")}
    for sensor, landmark_id, feature, mount in observations:
        if sensor not in counts:
            raise ValueError("Yalnız front ve rear sensörleri destekleniyor.")
        if type(landmark_id) is not int or not 0 <= landmark_id < estimator.landmark_count:
            raise ValueError("Geçersiz harita doğrusu kimliği.")
        counts[sensor] += 1

    result = {
        "matched_counts": counts,
        "matched_total": len(observations),
        "angle_limit_deg": angle_limit_deg,
        "angle_step_deg": angle_step_deg,
        "status": "insufficient_support",
        "nominal_nll": None,
        "models": [],
    }
    # İki sensörün de karşılaştırmaya katkı vermesini gerektiriyoruz.
    if not counts["front"] or not counts["rear"]:
        return result

    state = estimator.state.copy()
    covariance = estimator.P.copy()
    noise = estimator.line_R.copy()
    n = len(observations)
    R = np.kron(np.eye(n), noise)

    def score(model, angle_deg):
        angle = math.radians(angle_deg)
        candidate_robot = state[:4].copy()
        if model == "heading_offset":
            candidate_robot[2] = wrap_angle(candidate_robot[2] + angle)

        residuals = []
        jacobians = []
        for sensor, landmark_id, feature, nominal_mount in observations:
            mount = nominal_mount
            if model == f"{sensor}_yaw":
                mount = replace(mount, yaw=wrap_angle(mount.yaw + angle))

            index = 4 + 2 * landmark_id
            expected, H_robot, H_landmark = line_observation(
                candidate_robot, state[index:index + 2], mount
            )
            measurement = estimator._measurement(feature)
            if abs(wrap_angle(measurement[1] - expected[1])) > math.pi / 2:
                measurement[0] *= -1.0
                measurement[1] = wrap_angle(measurement[1] + math.pi)
            residual = measurement - expected
            residual[1] = wrap_angle(residual[1])
            H = np.zeros((2, len(state)))
            H[:, :4] = H_robot
            H[:, index:index + 2] = H_landmark
            residuals.append(residual)
            jacobians.append(H)

        residual = np.concatenate(residuals)
        H = np.vstack(jacobians)
        S = H @ covariance @ H.T + R
        S = (S + S.T) / 2.0
        L = np.linalg.cholesky(S)
        whitened = np.linalg.solve(L, residual)
        joint_nis = float(whitened @ whitened)
        logdet = float(2.0 * np.log(np.diag(L)).sum())
        nll = 0.5 * (joint_nis + logdet + 2 * n * math.log(2 * math.pi))
        if not math.isfinite(nll):
            raise ValueError("Sonlu olmayan hipotez skoru.")
        return nll, joint_nis

    nominal_nll, nominal_nis = score("nominal", 0.0)
    result.update({
        "status": "evaluated",
        "nominal_nll": nominal_nll,
        "nominal_joint_nis": nominal_nis,
        "residual_dimension": 2 * n,
    })

    count = math.floor(angle_limit_deg / angle_step_deg + 1e-12)
    if count > 200:
        raise ValueError("Açı arama ızgarası çok yoğun.")
    grid = [index * angle_step_deg for index in range(-count, count + 1)]
    # Sıfır açı, eşit skorlarda gereksiz bir düzeltmeden önce tercih edilir.
    grid.sort(key=lambda value: (abs(value), value))

    for model in ("front_yaw", "rear_yaw", "heading_offset"):
        candidates = [(score(model, angle), angle) for angle in grid]
        (nll, joint_nis), best_angle = min(
            candidates, key=lambda item: (item[0][0], abs(item[1]), item[1])
        )
        result["models"].append({
            "hypothesis": model,
            "best_offset_deg": best_angle,
            "nll": nll,
            "joint_nis": joint_nis,
            "nll_improvement_over_nominal": nominal_nll - nll,
            "at_search_edge": math.isclose(abs(best_angle), max(abs(v) for v in grid)),
        })
    return result
