import halide as hl
from ..wrapper import Wrapper
from ..math import sqrt, sum


def norm(w: Wrapper):
    return sqrt(sum(w**2))
