from __future__ import annotations
from collections.abc import Callable
from typing import Any, Sequence, Tuple
import halide as hl
import numpy as np
from dataclasses import dataclass
from .utils import var_from_index, vars_from_shape, tr


@dataclass
class Wrapper:
    shape: Tuple[Any]
    inner: hl.Func | hl.ImageParam

    def __getitem__(self, args):
        left_variables = tuple()
        right_variables = tuple()
        shape = tuple()
        for index, (arg, shaped) in enumerate(zip(args, self.shape)):
            var = var_from_index(index)
            if isinstance(arg, slice):
                left_variables += (var,)
                step = 1 if arg.step is None else arg.step

                if arg.start is None:
                    start = 0
                elif arg.start < 0:
                    start = shaped + arg.start
                else:
                    start = arg.start

                if arg.stop is None:
                    stop = shaped
                elif arg.stop < 0:
                    stop = shaped + arg.stop
                else:
                    stop = arg.stop

                extent = int(np.ceil(np.abs(stop - start) / np.max([np.abs(step), 1])))
                halide_step = 0 if extent == 1 else step
                right_variables += (halide_step * var + start,)
                shape += (extent,)
        f = hl.Func()
        f.__setitem__(tr(left_variables), self.inner.__getitem__(tr(right_variables)))
        return Wrapper(inner=f, shape=shape)

    def __add__(self, other) -> Wrapper:
        f = hl.Func()
        variables = vars_from_shape(self.shape)
        if isinstance(other, Wrapper):
            f[variables] = self.inner[variables] + other.inner[variables]
            shape = np.broadcast_shapes(self.shape, other.shape)
        else:
            f[variables] = self.inner[variables] + other
            shape = self.shape
        return Wrapper(inner=f, shape=shape)

    def __pow__(self, v) -> Wrapper:
        f = hl.Func()
        variables = vars_from_shape(self.shape)
        f[variables] = self.inner[variables] ** v
        return Wrapper(inner=f, shape=self.shape)

    def realize(self):
        return self.inner.realize(tr(self.shape))

    def to_halide(self):
        return self.realize()

    def to_numpy(self):
        return np.asanyarray(self.realize())

def array(values):
    np_array = np.array(values)
    return Wrapper(inner=hl.Buffer(np_array).copy(), shape=np_array.shape)


def wrap(values):
    return array(values)
