"""2B ortamın duvar ve dikdörtgen engel geometrisi."""

from dataclasses import dataclass
import math


Point = tuple[float, float]


@dataclass(frozen=True)
class Wall:
    """Metre cinsinden iki uç noktayla tanımlanan duvar."""

    start: Point
    end: Point

    def __post_init__(self):
        for point in (self.start, self.end):
            if len(point) != 2:
                raise ValueError(
                    "Bir nokta iki koordinat içermelidir."
                )

            if not all(math.isfinite(value) for value in point):
                raise ValueError(
                    "Duvar koordinatları sonlu olmalıdır."
                )

        if self.start == self.end:
            raise ValueError(
                "Duvarın uzunluğu sıfır olamaz."
            )

    @property
    def length(self) -> float:
        return math.hypot(
            self.end[0] - self.start[0],
            self.end[1] - self.start[1],
        )


@dataclass(frozen=True)
class Rectangle:
    """Alt-sol ve üst-sağ köşelerle tanımlanan dikdörtgen."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        values = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "Dikdörtgen koordinatları sonlu olmalıdır."
            )

        if self.x_min >= self.x_max:
            raise ValueError(
                "x_min, x_max değerinden küçük olmalıdır."
            )

        if self.y_min >= self.y_max:
            raise ValueError(
                "y_min, y_max değerinden küçük olmalıdır."
            )

    def walls(self) -> tuple[Wall, ...]:
        bottom_left = (self.x_min, self.y_min)
        bottom_right = (self.x_max, self.y_min)
        top_right = (self.x_max, self.y_max)
        top_left = (self.x_min, self.y_max)

        return (
            Wall(bottom_left, bottom_right),
            Wall(bottom_right, top_right),
            Wall(top_right, top_left),
            Wall(top_left, bottom_left),
        )


@dataclass(frozen=True)
class Environment:
    """Odanın sınırları ve sabit dikdörtgen engeller."""

    boundary: Rectangle
    obstacles: tuple[Rectangle, ...] = ()

    def __post_init__(self):
        for obstacle in self.obstacles:
            inside = (
                self.boundary.x_min < obstacle.x_min
                and obstacle.x_max < self.boundary.x_max
                and self.boundary.y_min < obstacle.y_min
                and obstacle.y_max < self.boundary.y_max
            )

            if not inside:
                raise ValueError(
                    "Engeller oda sınırlarının içinde "
                    "ve dış duvarlardan ayrı olmalıdır."
                )

    @property
    def walls(self) -> tuple[Wall, ...]:
        result = list(self.boundary.walls())

        for obstacle in self.obstacles:
            result.extend(obstacle.walls())

        return tuple(result)


def create_demo_environment() -> Environment:
    """Seçilmiş ilk deney ortamını üretir."""
    return Environment(
        boundary=Rectangle(0.0, 0.0, 10.0, 8.0),
        obstacles=(
            Rectangle(3.0, 2.0, 4.5, 4.0),
            Rectangle(6.5, 5.0, 8.0, 6.5),
        ),
    )