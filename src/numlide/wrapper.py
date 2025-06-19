from __future__ import annotations
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Optional, Sequence, Tuple
import halide as hl
import numpy as np
from dataclasses import dataclass

from numlide.matmul import matmul
from numlide.schedule import ScheduleStrategy
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

    def __post_init__(self):
        assert isinstance(self.shape, tuple), f"Wrapper shape must be tuple, found {type(self.shape)}"
        assert isinstance(self.inner, hl.Func), f"Wrapper inner must be hl.Func, found {type(self.inner)}"

    def __getitem__(self, args):
        try:
            len(args)
        except:
            args = [args]

        newaxis_count = args.count(None)

        if len(args) > len(self.shape) + newaxis_count:
            raise IndexError(
                f"IndexError: too many indices for array: array is {len(self.shape)}-dimensional, but {len(args)} were indexed"
            )

        missing_args = len(self.shape) - len(args)
        for _ in range(missing_args):
            args.append(slice(None))

        left_variables = tuple()
        right_variables = tuple()
        shape = tuple()

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

    def __setitem__(self, args, other):
        try:
            len(args)
        except:
            args = [args]

        if not isinstance(other, Wrapper):
            other = wrap(other)

        shape_diff = len(self.shape) - len(other.shape)
        if shape_diff < 0:
            raise ValueError(f"Cannot set item with shape {other.shape} into shape {self.shape}")

        if shape_diff > 0:
            other_args = []
            for _ in range(other.ndim):
                other_args.append(slice(None))
            for _ in range(shape_diff):
                other_args.append(None)
            other = other[tuple(other_args)]

        newaxis_count = args.count(None)

        if len(args) > len(self.shape) + newaxis_count:
            raise IndexError(
                f"IndexError: too many indices for array: array is {len(self.shape)}-dimensional, but {len(args)} were indexed"
            )

        missing_args = len(self.shape) - len(args)
        for _ in range(missing_args):
            args.append(slice(None))

        left_variables = tuple()
        right_variables = tuple()
        shape = tuple()

        for arg in args:
            left_index = len(left_variables)
            right_index = len(right_variables)
            var = var_from_index(left_index)
            if arg is None:
                raise NotImplementedError("Setting with None / newaxis not supported")
            elif isinstance(arg, int):
                left_variables += (arg,)
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
                rdom = hl.RDom([(start, extent)])
                left_variables += (halide_step * rdom,)
                right_variables += (halide_step * rdom,)
            else:
                raise NotImplementedError(f"Argument not supported: `{arg}` is of type `{type(arg)}`")
        other_value = other.inner[tr(right_variables)]
        self.inner[tr(left_variables)] = hl.cast(self.inner.type(), other_value)

    def _perform_operation(self, other, operation: _Operation) -> Wrapper:
        if not (isinstance(other, Wrapper) or isinstance(other, int) or isinstance(other, float)):
            other = wrap(other)
        f = hl.Func(str(operation).split(".")[1])
        if isinstance(other, Wrapper):
            variables = vars_from_shape(self.shape, zero_if_one=True)
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
            variables = vars_from_shape(self.shape)
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
        a = wrap(self)
        b = wrap(other)
        # if a.type() != b.type():
        #     raise ValueError(
        #         f"Matrices of different types are not supported by matmul yet, got {a.type()=} {b.type()=}"
        #     )

        a_buffer = a.realize()
        b_buffer = b.realize()

        result = matmul(a_buffer, b_buffer)

        return wrap(result)

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
        # self.inner.print_loop_nest()
        print(f"Realizing {self.shape}")
        result = self.inner.realize(tr(self.shape))
        print("Realizing done")
        return result

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
            and isinstance(inputs[0], (np.ndarray, float))
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
            if ufunc == np.exp:
                return math.exp(inputs[0])
            if ufunc == np.cos:
                return math.cos(inputs[0])
            if ufunc == np.sin:
                return math.sin(inputs[0])
            if ufunc == np.tan:
                return math.tan(inputs[0])
            if ufunc == np.tanh:
                return math.tanh(inputs[0])

        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        from . import math
        from . import manipulation

        if not all(issubclass(t, Wrapper) for t in types):
            return NotImplemented
        if func == np.mean:
            return math.mean(*args, **kwargs)
        if func == np.min:
            return math.min(*args, **kwargs)
        if func == np.max:
            return math.max(*args, **kwargs)
        if func == np.argmax:
            return math.argmax(*args, **kwargs)
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
        if func == np.split:
            return manipulation.split(*args, **kwargs)
        if func == np.array_split:
            return manipulation.array_split(*args, **kwargs)
        if func == np.swapaxes:
            return manipulation.swapaxes(*args, **kwargs)
        if func == np.hstack:
            return manipulation.hstack(*args, **kwargs)
        if func == np.vstack:
            return manipulation.vstack(*args, **kwargs)
        if func == np.concatenate:
            return manipulation.concatenate(*args, **kwargs)
        return NotImplemented

    def flatten(self):
        return wrap(self.to_numpy().flatten())

    @property
    def ndim(self):
        return len(self.shape)

    def transpose(self, axes: Optional[list[int] | tuple[int]] = None):
        axes = axes if axes else list(range(self.ndim)[::-1])
        vars = list(vars_from_shape(self.shape))
        vars_t = (vars[ax] for ax in axes)
        shape_t = tuple(self.shape[ax] for ax in axes)
        f = hl.Func(f"{self.inner.name()}_transpose")
        f[vars] = self.inner.__getitem__(list(vars_t))
        return Wrapper(shape=shape_t, inner=f)

    @property
    def T(self):
        return self.transpose()

    def type(self) -> hl.Type:
        return self.inner.type()


def array(values, name: Optional[str] = None):
    if isinstance(values, hl.Buffer):
        buffer = values
        variables = tuple()
        shape = tuple()
        for i in range(buffer.dimensions()):
            variables += (var_from_index(i),)
            shape += (buffer.dim(i).extent(),)
        shape = tr(shape)
        print(f"Buffer shape {shape=}")
    else:
        np_array = np.array(values)
        buffer_name = name if name else "np_array"
        buffer = hl.Buffer(np_array, name=buffer_name).copy()
        variables = vars_from_shape(np_array.shape)
        shape = np_array.shape
        print(f"Array shape {shape=}")

    inner = hl.Func("array")
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

    return Wrapper(inner=inner, shape=shape)


def wrap(values, name: Optional[str] = None):
    return array(values, name=name)
