from typing import Optional
import halide as hl
from .utils import calculate_extent, var_from_index
from .wrapper import Wrapper


def arange(start_or_stop: int, stop: Optional[int] = None, step: Optional[int] = None):
    if stop is None:
        stop = start_or_stop
        start = 0
    else:
        start = start_or_stop
        stop = stop

    step = 1 if step is None else step

    f = hl.Func("arange")
    v = var_from_index(0)
    f[v] = start + v * step
    extent = calculate_extent(start, stop, step)
    shape = tuple((extent,))
    return Wrapper(shape=shape, inner=f)
