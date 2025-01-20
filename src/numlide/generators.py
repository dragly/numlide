from numlide.utils import halide_type, vars_from_shape
from .wrapper import Wrapper
from typing import Any, Sequence
import halide as hl
import numpy as np


def ones(shape: Sequence[int], dtype: Any = None) -> Wrapper:
    if dtype is None:
        dtype = np.float64
    f = hl.Func("ones")
    vars = vars_from_shape(shape)
    f[vars] = hl.cast(halide_type(dtype), 1)
    return Wrapper(inner=f, shape=shape)


def zeros(shape: Sequence[int], dtype: Any = None) -> Wrapper:
    if dtype is None:
        dtype = np.float64
    f = hl.Func("zeros")
    vars = vars_from_shape(shape)
    f[vars] = hl.cast(halide_type(dtype), 0)
    return Wrapper(inner=f, shape=shape)
