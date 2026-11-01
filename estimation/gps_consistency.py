"""GPS'ten bağımsız konum referansıyla tutarsızlık izleme.

GPS hatası ile referansın küresel kayması her durumda
ayırt edilemez. Kesin kaynak teşhisi üretmez.
"""

import math

import numpy as np


class GPSConsistencyMonitor:
    def __init__(
        self,
        threshold=9.21034,
        consecutive=3,
    ):
        if (
            not math.isfinite(threshold)
            or threshold <= 0
            or type(consecutive) is not int
            or consecutive < 2
        ):
            raise ValueError(
                "Geçersiz GPS izleyici ayarı."
            )

        self.threshold = float(threshold)
        self.required = consecutive
        self.streak = 0
        self.confirmed = False
        self.confirmation_time_s = None
        self.last_time = None
        self.history = []

    def observe(self, estimator, packet):
        time = packet.time_s

        if (
            self.last_time is not None
            and time <= self.last_time
        ):
            raise ValueError(
                "GPS zamanları artmalı."
            )

        if not math.isclose(
            time,
            estimator.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "GPS ve referans zamanı farklı."
            )

        row = {
            "time_s": time,
            "available": packet.available,
            "nis": None,
            "residual_x_m": None,
            "residual_y_m": None,
            "instant_reject": False,
        }

        if not packet.available:
            self.streak = 0

        else:
            residual = np.array([
                packet.x_m,
                packet.y_m,
            ]) - estimator.state[:2]

            covariance = (
                np.asarray(
                    estimator.P[:2, :2],
                    dtype=float,
                )
                + np.eye(2) * max(
                    packet.nominal_variance_m2,
                    1e-8,
                )
            )

            if (
                not np.isfinite(residual).all()
                or not np.isfinite(covariance).all()
            ):
                raise ValueError(
                    "Sonlu olmayan GPS referansı."
                )

            np.linalg.cholesky(covariance)

            nis = float(
                residual
                @ np.linalg.solve(
                    covariance,
                    residual,
                )
            )

            reject = nis > self.threshold

            self.streak = (
                self.streak + 1
                if reject
                else 0
            )

            if (
                self.streak >= self.required
                and not self.confirmed
            ):
                self.confirmed = True
                self.confirmation_time_s = time

            row.update(
                nis=nis,
                residual_x_m=float(residual[0]),
                residual_y_m=float(residual[1]),
                instant_reject=bool(reject),
            )

        row.update(
            consecutive_exceedances=self.streak,
            confirmed=self.confirmed,
            confirmation_time_s=(
                self.confirmation_time_s
            ),
        )

        self.last_time = time
        self.history.append(row)

        return row


def joint_status(
    geometry_suspect,
    gps_confirmed,
):
    if geometry_suspect and gps_confirmed:
        return "multiple_inconsistencies"

    if geometry_suspect:
        return "geometric_inconsistency"

    if gps_confirmed:
        return "gps_reference_inconsistency"

    return "no_confirmed_inconsistency"