from __future__ import annotations
from collections.abc import Callable
from typing import Any, Sequence, Tuple
import halide as hl
import numpy as np
from dataclasses import dataclass


def var_from_index(index: int):
    return hl.Var(f"v{index}")


def tr(values: Sequence[hl.Var | hl.Expr]) -> Tuple[hl.Var | hl.Expr]:
    return tuple(reversed(values))


def vars_from_shape(shape: Tuple[int]) -> Tuple[hl.Var]:
    variables = tuple()
    for i in range(len(shape)):
        variables += (var_from_index(i),)
    return tr(variables)


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
                start = 0 if arg.start is None else arg.start
                stop = shaped if arg.stop is None else arg.stop
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
            shape=np.broadcast_shapes(self.shape, other.shape)
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


def apply(w: Wrapper, f: Callable[[hl.Expr], hl.Expr]) -> Wrapper:
    func = hl.Func()
    variables = vars_from_shape(w.shape)
    func[variables] = f(w.inner[variables])
    return Wrapper(inner=func, shape=w.shape)


def sin(w: Wrapper) -> Wrapper:
    return apply(w, hl.sin)

def cos(w: Wrapper) -> Wrapper:
    return apply(w, hl.cos)

def tan(w: Wrapper) -> Wrapper:
    return apply(w, hl.tan)

def sqrt(w: Wrapper) -> Wrapper:
    return apply(w, hl.sqrt)

def sum(w: Wrapper) -> Wrapper:
    f = hl.Func()
    f[()] = 0.0
    rdom_elements = list()
    for extent in w.shape:
        rdom_elements.append((0, extent))

    rdom = hl.RDom(tr(rdom_elements))
    rdom_accesors = []
    for i in range(rdom.dimensions()):
        rdom_accesors.append(rdom[i])
    f[()] += w.inner[rdom_accesors]
    return Wrapper(inner=f, shape=tuple())


def array(values):
    np_array = np.array(values)
    return Wrapper(inner=hl.Buffer(np_array).copy(), shape=np_array.shape)
