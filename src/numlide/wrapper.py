from __future__ import annotations
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Sequence, Tuple
import halide as hl
import numpy as np
from dataclasses import dataclass
from .utils import calculate_extent, var_from_index, vars_from_shape, tr
from itertools import zip_longest


class _Operation(Enum):
    add = auto()
    sub = auto()
    mul = auto()
    matmul = auto()
    truediv = auto()
    floordiv = auto()
    mod = auto()
    pow_ = auto()
    lt = auto()
    gt = auto()
    le = auto()
    ge = auto()
    eq = auto()
    ne = auto()


@dataclass
class Wrapper:
    shape: Tuple[Any]
    inner: hl.Func

    def __getitem__(self, args):
        left_variables = tuple()
        right_variables = tuple()
        shape = tuple()

        try:
            len(args)
        except:
            args = [args]

        for arg in args:
            left_index = len(left_variables)
            right_index = len(right_variables)
            var = var_from_index(left_index)
            if arg is None:
                left_variables += (var,)
                shape += (1,)
            elif isinstance(arg, int):
                right_variables += (arg,)
            elif isinstance(arg, slice):
                step = 1 if arg.step is None else arg.step

                shaped = self.shape[right_index]

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

                extent = calculate_extent(start, stop, step)
                halide_step = 0 if extent == 1 else step
                left_variables += (var,)
                right_variables += (halide_step * var + start,)
                shape += (extent,)
            else:
                raise NotImplementedError(f"Argument not supported: `{arg}` is of type `{type(arg)}`")
        f = hl.Func("getitem")
        f[tr(left_variables)] = self.inner[tr(right_variables)]
        return Wrapper(inner=f, shape=shape)

    def _perform_operation(self, other, operation: _Operation) -> Wrapper:
        if not (isinstance(other, Wrapper) or isinstance(other, int) or isinstance(other, float)):
            other = wrap(other)
        f = hl.Func(str(operation).split(".")[1])
        variables = vars_from_shape(self.shape, zero_if_one=True)
        if isinstance(other, Wrapper):
            broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
            new_variables = vars_from_shape(broadcast_shape)
            if len(other.shape) == 1 and len(broadcast_shape) > 1:
                # custom broadcasting case where the last dimension
                # will be applied against the other
                other_variables = [new_variables[0]]
            else:
                other_variables = vars_from_shape(other.shape, zero_if_one=True)

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
                case _Operation.lt:
                    # TODO: Remove cast and ensure other functions, such as sum, do the casting
                    f[new_variables] = hl.i32(self.inner[variables] < other.inner[other_variables])
                case _Operation.gt:
                    f[new_variables] = hl.i32(self.inner[variables] > other.inner[other_variables])
                case _Operation.le:
                    f[new_variables] = hl.i32(self.inner[variables] <= other.inner[other_variables])
                case _Operation.ge:
                    f[new_variables] = hl.i32(self.inner[variables] >= other.inner[other_variables])
                case _Operation.eq:
                    f[new_variables] = hl.i32(self.inner[variables] == other.inner[other_variables])
                case _Operation.ne:
                    f[new_variables] = hl.i32(self.inner[variables] != other.inner[other_variables])
                case _:
                    raise RuntimeError(f"Operation not supported: {operation}")
        else:
            a = self.inner[variables]
            if self.inner.type() == hl.Float(32):
                b = hl.f32(other)
            elif self.inner.type() == hl.Float(64):
                b = hl.f64(other)
            else:
                b = other

            match operation:
                case _Operation.add:
                    f[variables] = a + b
                case _Operation.sub:
                    f[variables] = a - b
                case _Operation.mul:
                    f[variables] = a * b
                case _Operation.truediv:
                    f[variables] = a / b
                case _Operation.floordiv:
                    f[variables] = a // b
                case _Operation.mod:
                    f[variables] = a % b
                case _Operation.pow_:
                    f[variables] = a**b
                case _Operation.lt:
                    f[variables] = a < b
                case _Operation.gt:
                    f[variables] = a > b
                case _Operation.le:
                    f[variables] = a <= b
                case _Operation.ge:
                    f[variables] = a >= b
                case _Operation.eq:
                    f[variables] = a == b
                case _Operation.ne:
                    f[variables] = a != b
                case _:
                    raise RuntimeError(f"Operation not supported: {operation}")
            broadcast_shape = self.shape
        return Wrapper(inner=f, shape=broadcast_shape)

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

    def __matmul__(self, other) -> Wrapper:
        if not (isinstance(other, Wrapper)):
            other = wrap(other)
        f = hl.Func("matmul")
        variables = vars_from_shape(self.shape)
        r = hl.RDom([(0, self.shape[-1])])
        f[variables] = hl.cast(self.inner.type(), 0)
        f[variables] += self.inner[(r,) + variables[1:]] * other.inner[tuple(variables[:-1]) + (r,)]
        new_shape = self.shape[:-1] + other.shape[1:]
        return Wrapper(inner=f, shape=new_shape)

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

    def __lt__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.lt)

    def __le__(self, other) -> Wrapper:
        return self._perform_operation(other, _Operation.le)

    def realize(self):
        return self.inner.realize(tr(self.shape))

    def to_halide(self):
        return self.realize()

    def to_numpy(self):
        return np.asanyarray(self.realize())

    def print_loop_nest(self):
        self.inner.print_loop_nest()

    def __str__(self):
        return str(np.asanyarray(self.realize()))

    def __repr__(self):
        return repr(np.asanyarray(self.realize()))

    def __len__(self):
        if len(self.shape) > 0:
            return self.shape[0]
        return 0

    def __contains__(self, v):
        return np.asanyarray(self.realize()).__contains__(v)

    def __iter__(self):
        return np.asanyarray(self.realize()).__iter__()

    def __abs__(self) -> Wrapper:
        vars = vars_from_shape(self.shape)
        f = hl.Func(f"{self.inner.name()}_abs")
        f[vars] = hl.abs(self.inner[vars])
        return Wrapper(shape=self.shape, inner=f)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from . import math

        if (
            method == "__call__"
            and len(inputs) == 2
            and isinstance(inputs[0], np.ndarray)
            and isinstance(inputs[1], Wrapper)
        ):
            if ufunc == np.add:
                return inputs[1] + inputs[0]
            if ufunc == np.less_equal:
                return wrap(inputs[0]) <= inputs[1]
            if ufunc == np.greater:
                return wrap(inputs[0]) > inputs[1]
            if ufunc == np.multiply:
                return wrap(inputs[0]) * inputs[1]
        if method == "__call__" and len(inputs) == 1 and isinstance(inputs[0], Wrapper):
            if ufunc == np.sqrt:
                return math.sqrt(inputs[0])

        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        from . import math

        if not all(issubclass(t, Wrapper) for t in types):
            return NotImplemented
        if func == np.mean:
            return math.mean(*args, **kwargs)
        if func == np.min:
            return math.min(*args, **kwargs)
        if func == np.max:
            return math.max(*args, **kwargs)
        if func == np.iscomplexobj:
            return False
        if func == np.abs:
            return math.abs(*args, **kwargs)
        if func == np.sqrt:
            return math.sqrt(*args, **kwargs)
        if func == np.sum:
            return math.sum(*args, **kwargs)
        if func == np.var:
            return math.var(*args, **kwargs)
        return NotImplemented


def array(values):
    np_array = np.array(values)
    buffer = hl.Buffer(np_array).copy()
    inner = hl.Func("array")
    variables = vars_from_shape(np_array.shape)
    if len(variables) == 0:
        if buffer.type() == hl.Float(32):
            other = hl.f32(buffer[variables])
        elif buffer.type() == hl.Float(64):
            other = hl.f64(buffer[variables])
        else:
            other = buffer[variables]
    else:
        other = buffer[variables]
    inner[variables] = other
    return Wrapper(inner=inner, shape=np_array.shape)


def wrap(values):
    return array(values)
