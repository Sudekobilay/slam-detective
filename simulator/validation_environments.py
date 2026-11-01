"""EXP-42 için sonuçlara bakılmadan tanımlanan iki ek test geometrisi."""
from simulator.environment import Environment, Rectangle


def validation_environment(name):
    if name == 'corridor':
        return Environment(Rectangle(0., 1.8, 10., 4.4))
    if name == 'offset_obstacles':
        return Environment(Rectangle(0., 0., 9., 7.), (
            Rectangle(2.6, 4., 4., 6.2),
            Rectangle(5., .8, 6.5, 2.5),
        ))
    raise ValueError('Tanımsız doğrulama ortamı.')
