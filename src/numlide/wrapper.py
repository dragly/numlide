from __future__ import annotations
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Optional, Sequence, Tuple
import halide as hl
import numpy as np
from dataclasses import dataclass

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

    def __matmul__(
        self,
        other,
        schedule_strategy=ScheduleStrategy.auto,
    ) -> Wrapper:
        if not (isinstance(other, Wrapper)):
            other = wrap(other)
        a = wrap(self.to_numpy())
        b = wrap(other.to_numpy())
        matmul = hl.Func("matmul_impl")
        variables = vars_from_shape(a.shape)
        matrix_size = a.shape[-1]
        k = hl.RDom([(0, matrix_size)])
        matmul[variables] = hl.cast(a.inner.type(), 0)
        matmul[variables] += a.inner[(k,) + variables[1:]] * b.inner[tuple(variables[:-1]) + (k,)]
        output = hl.Func("matmul")
        output[variables] = matmul[variables]
        output_size = a.shape[0]
        if schedule_strategy == ScheduleStrategy.auto:
            x = variables[0]
            y = variables[1]
            xy = hl.Var("xy")
            xi = hl.Var("xi")
            yi = hl.Var("yi")
            xo = hl.Var("xo")
            yo = hl.Var("yo")
            yii = hl.Var("yii")
            if output_size > 32 and matrix_size > 32:
                # schedule copied from
                # https://github.com/halide/Halide/blob/bf65d521d69d75c0ffa9459cdf797886b1bc77e2/test/performance/matrix_multiplication.cpp
                target = hl.get_jit_target_from_environment()
                vec = target.natural_vector_size(a.inner.type())
                inner_tile_x = 3 * vec
                inner_tile_y = 8
                tile_y = output_size // 4
                tile_k = matrix_size // 16
                output.tile(
                    x,
                    y,
                    xo,
                    yo,
                    xi,
                    yi,
                    inner_tile_x,
                    tile_y,
                ).split(
                    yi,
                    yi,
                    yii,
                    inner_tile_y,
                ).vectorize(
                    xi, vec
                ).unroll(xi).unroll(yii).fuse(xo, yo, xy).parallel(xy)
                ko = hl.RVar("ko")
                ki = hl.RVar("ki")
                z = hl.Var("z")
                matmul.update().split(
                    k,
                    ko,
                    ki,
                    tile_k,
                )
                intm = matmul.update().rfactor(ko, z)

                intm.compute_at(matmul, y).vectorize(x, vec).unroll(x).unroll(y)

                intm.update(0).reorder(x, y, ki).vectorize(x, vec).unroll(x).unroll(y)

                matmul.compute_at(output, xy).vectorize(x, vec).unroll(x)

                matmul.update().split(
                    y,
                    y,
                    yi,
                    inner_tile_y,
                ).reorder(x, yi, y, ko).vectorize(
                    x,
                    vec,
                ).unroll(
                    x
                ).unroll(yi)

                output.bound(x, 0, output_size).bound(y, 0, output_size)
                output.compute_root()
            else:
                output.tile(x, y, xo, yo, xi, yi, 4, 4).vectorize(xi, 4)
                output.compute_root()
        new_shape = a.shape[:-1] + b.shape[1:]
        return Wrapper(inner=output, shape=new_shape)

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
        # print(f"Realizing {self.shape}")
        result = self.inner.realize(tr(self.shape))
        # print("Realizing done")
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
