from __future__ import annotations
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Sequence, Tuple
import halide as hl
import numpy as np
from dataclasses import dataclass
from .utils import var_from_index, vars_from_shape, tr


class _Operation(Enum):
    add = auto()
    sub = auto()
    mul = auto()
    truediv = auto()
    floordiv = auto()
    mod = auto()
    pow_ = auto()


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
        f = hl.Func("getitem")
        f.__setitem__(tr(left_variables), self.inner.__getitem__(tr(right_variables)))
        return Wrapper(inner=f, shape=shape)

    def _perform_operation(self, other, operation: _Operation) -> Wrapper:
        if not isinstance(other, Wrapper) or isinstance(other, int) or isinstance(other, float):
            other = wrap(other)
        f = hl.Func("add")
        variables = vars_from_shape(self.shape)
        if isinstance(other, Wrapper):
            other_variables = vars_from_shape(other.shape)
            new_variables = variables + other_variables
            match operation:
                case _Operation.add:
                    f[new_variables] = self.inner[variables] + other.inner[other_variables]
                case _Operation.sub:
                    f[new_variables] = self.inner[variables] - other.inner[other_variables]
                case _Operation.mul:
                    f[new_variables] = self.inner[variables] * other.inner[other_variables]
                case _Operation.truediv:
                    f[new_variables] = self.inner[variables] / other.inner[other_variables]
                case _Operation.floordiv:
                    f[new_variables] = self.inner[variables] // other.inner[other_variables]
                case _Operation.mod:
                    f[new_variables] = self.inner[variables] % other.inner[other_variables]
                case _Operation.pow_:
                    f[new_variables] = self.inner[variables] ** other.inner[other_variables]
                case _:
                    raise RuntimeError(f"Operation not supported: {operation}")
            shape = np.broadcast_shapes(self.shape, other.shape)
        else:
            match operation:
                case _Operation.add:
                    f[variables] = self.inner[variables] + other
                case _Operation.sub:
                    f[variables] = self.inner[variables] - other
                case _Operation.mul:
                    f[variables] = self.inner[variables] * other
                case _Operation.truediv:
                    f[variables] = self.inner[variables] / other
                case _Operation.floordiv:
                    f[variables] = self.inner[variables] // other
                case _Operation.mod:
                    f[variables] = self.inner[variables] % other
                case _Operation.pow_:
                    f[variables] = self.inner[variables] ** other
                case _:
                    raise RuntimeError(f"Operation not supported: {operation}")
            shape = self.shape
        return Wrapper(inner=f, shape=shape)

    def __add__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.add)

    def __radd__(self, other) -> Wrapper:
        return self + other

    def __sub__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.sub)

    def __rsub__(self, other) -> Wrapper:
        return wrap(other) - self

    def __mul__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.mul)

    def __rmul__(self, other) -> Wrapper:
        return self * other

    def __truediv__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.truediv)

    def __rtruediv__(self, other) -> Wrapper:
        return wrap(other) / self

    def __floordiv__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.floordiv)

    def __rfloordiv__(self, other) -> Wrapper:
        return wrap(other) // self

    def __mod__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.mod)

    def __rmod__(self, other) -> Wrapper:
        return wrap(other) % self

    def __pow__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.pow_)

    def __rpow__(self, other) -> Wrapper:
        return wrap(other) ** self

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
