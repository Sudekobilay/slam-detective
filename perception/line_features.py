"""Sıralı LiDAR noktalarından bölerek doğru parçası çıkarımı."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LineConfig:
    max_point_error_m: float = 0.02
    max_gap_m: float = 0.5
    min_points: int = 5
    min_length_m: float = 0.15

    def __post_init__(self):
        positive_values = (
            self.max_point_error_m,
            self.max_gap_m,
            self.min_length_m,
        )

        if not all(
            math.isfinite(value) and value > 0.0
            for value in positive_values
        ):
            raise ValueError("Geometrik eşikler pozitif olmalıdır.")

        if type(self.min_points) is not int or self.min_points < 3:
            raise ValueError("En az üç destek noktası gerekli.")


@dataclass(frozen=True)
class LineFeature:
    # Birim normal ve uzaklık: normal · point = rho.
    normal: tuple[float, float]
    rho_m: float

    # Doğruya izdüşürülmüş uç noktalar.
    start: tuple[float, float]
    end: tuple[float, float]

    first_beam: int
    last_beam: int
    support_count: int
    length_m: float
    max_point_error_m: float


def extract_lines(points, config=None):
    """
    Sensör koordinatındaki sıralı noktaları işler.

    None noktaları ve büyük komşuluk boşlukları grupları ayırır.
    Dünya haritası, gerçek robot pozu veya nesne kimliği almaz.
    """
    config = config if config is not None else LineConfig()

    groups = []
    current = []

    for beam_index, point in enumerate(points):
        if point is None:
            if current:
                groups.append(current)
                current = []
            continue

        if (
            len(point) != 2
            or not all(math.isfinite(value) for value in point)
        ):
            raise ValueError("Noktalar sonlu iki koordinat içermeli.")

        point = (float(point[0]), float(point[1]))

        if current:
            previous_point = current[-1][1]

            if math.dist(previous_point, point) > config.max_gap_m:
                groups.append(current)
                current = []

        current.append((beam_index, point))

    if current:
        groups.append(current)

    features = []

    def split_and_fit(group):
        if len(group) < config.min_points:
            return

        coordinates = np.array(
            [item[1] for item in group],
            dtype=float,
        )
        center = coordinates.mean(axis=0)

        # Dik uzaklıkları en aza indiren doğru: TLS/SVD.
        _, _, Vt = np.linalg.svd(
            coordinates - center,
            full_matrices=False,
        )
        tangent = Vt[0]
        normal = Vt[1]
        rho = float(normal @ center)

        if rho < 0.0:
            normal = -normal
            rho = -rho

        errors = np.abs(coordinates @ normal - rho)

        projections = (coordinates - center) @ tangent
        start = center + float(projections.min()) * tangent
        end = center + float(projections.max()) * tangent
        length = float(np.linalg.norm(end - start))

        if float(errors.max()) <= config.max_point_error_m:
            if length < config.min_length_m:
                return

            features.append(LineFeature(
                normal=(float(normal[0]), float(normal[1])),
                rho_m=rho,
                start=(float(start[0]), float(start[1])),
                end=(float(end[0]), float(end[1])),
                first_beam=group[0][0],
                last_beam=group[-1][0],
                support_count=len(group),
                length_m=length,
                max_point_error_m=float(errors.max()),
            ))
            return

        # Bölme noktasını iki tarafta paylaşarak köşeyi koru.
        if len(group) < 2 * config.min_points - 1:
            return

        split_index = int(np.argmax(errors))
        split_index = max(
            config.min_points - 1,
            min(split_index, len(group) - config.min_points),
        )

        split_and_fit(group[:split_index + 1])
        split_and_fit(group[split_index:])

    for group in groups:
        split_and_fit(group)

    return features