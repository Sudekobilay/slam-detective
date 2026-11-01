import math

import numpy as np

from estimation.line_ekf_slam import LineEKFSLAM
from estimation.pose_ekf import PoseEKF


class GPSGateMixin:
    """GPS konum ölçümleri için NIS tabanlı güncelleme kapısı."""

    def __init__(self, *args, gps_nis_threshold=9.21034, **kwargs):
        threshold = float(gps_nis_threshold)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("GPS NIS eşiği pozitif ve sonlu olmalı.")

        super().__init__(*args, **kwargs)
        self.gps_nis_threshold = threshold
        self.gps_gate_log = []
        self.gps_rejected = 0

    def update_gps(self, measurement):
        time_s = measurement.time_s

        if not math.isclose(
            time_s, self.time_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("GPS zamanı kestiriciyle uyuşmuyor.")

        if self._last_gps_time is not None and time_s <= self._last_gps_time:
            raise ValueError("GPS ölçümü tekrarlandı veya sırası bozuk.")

        if not measurement.available:
            return super().update_gps(measurement)

        residual = np.array([
            measurement.x_m - self.state[0],
            measurement.y_m - self.state[1],
        ])

        variance = max(
            measurement.nominal_variance_m2,
            self.gps_variance_floor,
        )

        # H yalnız x ve y bileşenlerini seçtiği için:
        innovation_covariance = (
            self.P[:2, :2] + np.eye(2) * variance
        )

        nis = float(
            residual
            @ np.linalg.solve(innovation_covariance, residual)
        )
        accepted = nis <= self.gps_nis_threshold

        self.gps_gate_log.append({
            "time_s": time_s,
            "nis": nis,
            "threshold": self.gps_nis_threshold,
            "accepted": accepted,
        })

        if not accepted:
            # Ölçüm işlendi; durum ve kovaryans değiştirilmedi.
            self._last_gps_time = time_s
            self.gps_rejected += 1
            return None

        return super().update_gps(measurement)


class GatedPoseEKF(GPSGateMixin, PoseEKF):
    pass


class GatedLineEKFSLAM(GPSGateMixin, LineEKFSLAM):
    pass