"""Offline diagnostic movement metrics."""
import math

def movement_metrics(truth, commands, trigger, end):
    """Çevrimdışı hareket maliyeti; alarm anı başlangıca dahildir."""
    if trigger is None:
        return {
            key: None
            for key in (
                "diagnostic_distance_m",
                "diagnostic_absolute_turn_rad",
                "changed_command_duration_s",
                "diagnostic_duration_s",
            )
        }

    points = [
        point
        for point in truth
        if trigger - 1e-9 <= point["time_s"] <= end + 1e-9
    ]

    distance = sum(
        math.hypot(
            second["x_m"] - first["x_m"],
            second["y_m"] - first["y_m"],
        )
        for first, second in zip(points, points[1:])
    )

    turn = sum(
        abs(
            math.atan2(
                math.sin(second["theta_rad"] - first["theta_rad"]),
                math.cos(second["theta_rad"] - first["theta_rad"]),
            )
        )
        for first, second in zip(points, points[1:])
    )

    duration = sum(
        min(
            commands[index + 1]["interval_start_s"]
            if index + 1 < len(commands)
            else truth[-1]["time_s"],
            end,
        ) - command["interval_start_s"]
        for index, command in enumerate(commands)
        if (
            command["applied"]
            and command["changed_from_normal"]
            and trigger - 1e-9
            <= command["interval_start_s"]
            < end - 1e-9
        )
    )

    return {
        "diagnostic_distance_m": distance,
        "diagnostic_absolute_turn_rad": turn,
        "changed_command_duration_s": duration,
        "diagnostic_duration_s": (
            max(0.0, points[-1]["time_s"] - trigger)
            if points
            else 0.0
        ),
    }
