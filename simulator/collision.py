"""Dairesel robot gövdesi için geometrik çarpışma kontrolü."""

import math

from simulator.environment import Environment, Rectangle
from simulator.robot import DifferentialDriveRobot, Pose


def signed_rectangle_distance(
    x: float,
    y: float,
    rectangle: Rectangle,
) -> float:
    """
    Noktanın dolu dikdörtgene işaretli uzaklığı.

    Dışarıda pozitif, sınırda sıfır, içeride negatiftir.
    """
    dx = max(
        rectangle.x_min - x,
        0.0,
        x - rectangle.x_max,
    )
    dy = max(
        rectangle.y_min - y,
        0.0,
        y - rectangle.y_max,
    )

    if dx > 0.0 or dy > 0.0:
        return math.hypot(dx, dy)

    return -min(
        x - rectangle.x_min,
        rectangle.x_max - x,
        y - rectangle.y_min,
        rectangle.y_max - y,
    )


def body_clearance(
    pose: Pose,
    environment: Environment,
    radius: float = 0.35,
) -> float:
    """
    Robot gövdesinin çevreye en küçük açıklığı.

    Pozitif: ayrık.
    Sıfır: temas.
    Negatif: örtüşme veya oda dışına taşma.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            "Gövde yarıçapı pozitif ve sonlu olmalıdır."
        )

    boundary = environment.boundary

    clearance = min(
        pose.x - boundary.x_min,
        boundary.x_max - pose.x,
        pose.y - boundary.y_min,
        boundary.y_max - pose.y,
    ) - radius

    for obstacle in environment.obstacles:
        clearance = min(
            clearance,
            signed_rectangle_distance(
                pose.x, pose.y, obstacle
            ) - radius,
        )

    return clearance


def try_step(
    robot: DifferentialDriveRobot,
    environment: Environment,
    left_rate: float,
    right_rate: float,
    dt: float,
    radius: float = 0.35,
    max_center_travel: float = 0.02,
) -> bool:
    """
    Hareket güvenli bulunursa uygular ve True döndürür.

    Reddedilirse robotun pozu değişmez.
    Kontrol tüm komut aralığını kapsar.
    """
    if not all(
        math.isfinite(value)
        for value in (left_rate, right_rate, dt)
    ):
        raise ValueError("Komut değerleri sonlu olmalıdır.")

    if dt < 0.0:
        raise ValueError("Süre negatif olamaz.")

    if (
        not math.isfinite(max_center_travel)
        or max_center_travel <= 0.0
    ):
        raise ValueError(
            "Kontrol aralığı pozitif ve sonlu olmalıdır."
        )

    initial_pose = robot.pose

    if body_clearance(initial_pose, environment, radius) <= 0.0:
        raise ValueError(
            "Robot başlangıçta engelle temas ediyor "
            "veya geçersiz konumda."
        )

    geometry = robot.geometry

    left_velocity = geometry.left_radius * left_rate
    right_velocity = geometry.right_radius * right_rate
    velocity = (left_velocity + right_velocity) / 2.0

    travel = abs(velocity) * dt

    if not math.isfinite(travel):
        raise ValueError("Hareket mesafesi sonlu olmalıdır.")

    number_of_checks = max(
        1,
        math.ceil(travel / max_center_travel),
    )

    if number_of_checks > 100_000:
        raise ValueError(
            "Komut çok uzun; daha kısa zaman adımlarına böl."
        )

    interval = dt / number_of_checks

    # Her aralıkta merkez, orta noktadan en fazla
    # yarım aralık boyunca aldığı yol kadar uzaklaşabilir.
    movement_bound = abs(velocity) * interval / 2.0

    for index in range(number_of_checks):
        midpoint_time = (index + 0.5) * interval

        probe = DifferentialDriveRobot(
            geometry=geometry,
            initial_pose=initial_pose,
        )
        midpoint_pose = probe.step(
            left_rate, right_rate, midpoint_time
        )

        clearance = body_clearance(
            midpoint_pose, environment, radius
        )

        if clearance <= movement_bound + 1e-12:
            return False

    trial = DifferentialDriveRobot(
        geometry=geometry,
        initial_pose=initial_pose,
    )
    candidate_pose = trial.step(left_rate, right_rate, dt)

    if body_clearance(candidate_pose, environment, radius) <= 0.0:
        return False

    # Ancak bütün kontroller geçerse gerçek robotu güncelle.
    robot.pose = candidate_pose
    return True