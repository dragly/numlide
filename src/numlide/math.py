from typing import Any, Optional, Tuple
import halide as hl
from collections.abc import Callable

from .schedule import ScheduleStrategy
from .wrapper import Wrapper, wrap
from .utils import var_from_index, vars_from_shape, tr
import numpy as np


def apply(w: Wrapper, f: Callable[[hl.Expr], hl.Expr]) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)
    func = hl.Func(f.__name__)
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


def _deduce_axis(
    wrapper: Wrapper,
    axis: Optional[int | Tuple[int]],
) -> Tuple[int, ...]:
    if axis is None:
        return tuple(i for i in range(len(wrapper.shape)))
    elif isinstance(axis, int):
        return tuple((axis,))
    return axis


def _reduce(
    w: Wrapper,
    impl: Callable[[Any, Any], hl.Func],
    axis: Optional[int | Tuple[int]],
    schedule_strategy=ScheduleStrategy.auto,
):
    axis = _deduce_axis(w, axis)

    rdom_elements = list()
    for i, extent in enumerate(w.shape):
        if i in axis:
            rdom_elements.append((0, extent))

    shape = tuple()
    next_rdom_element = 0
    rdom = hl.RDom(tr(rdom_elements))
    right_variables = []
    left_variables = []
    for i in tr(range(len(w.shape))):
        if i in axis:
            right_variables.append(rdom[next_rdom_element])
            next_rdom_element += 1
        else:
            var = var_from_index(i)
            right_variables.append(var)
            left_variables.append(var)
            shape += (w.shape[i],)

    shape = tr(shape)

    f = impl(left_variables, right_variables)

    if schedule_strategy == ScheduleStrategy.auto:
        f.compute_root()

    return Wrapper(inner=f, shape=shape)


def sum(w: Wrapper, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    def impl(left_variables, right_variables) -> hl.Func:
        f = hl.Func("sum")
        f[left_variables] = hl.cast(w.inner.type(), 0)
        f[left_variables] += w.inner[right_variables]
        return f

    return _reduce(w, impl=impl, axis=axis, schedule_strategy=schedule_strategy)


def min(w: Wrapper, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    def impl(left_variables, right_variables) -> hl.Func:
        f = hl.Func("min")
        f[left_variables] = hl.minimum(w.inner[right_variables])
        return f

    return _reduce(w, impl=impl, axis=axis, schedule_strategy=schedule_strategy)


def max(w: Wrapper, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    def impl(left_variables, right_variables) -> hl.Func:
        f = hl.Func("max")
        f[left_variables] = hl.maximum(w.inner[right_variables])
        return f

    return _reduce(w, impl=impl, axis=axis, schedule_strategy=schedule_strategy)


def mean(w: Wrapper, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    axis = _deduce_axis(w, axis)
    summed_element_count = np.prod(np.array(w.shape)[list(axis)])

    f_f64 = hl.Func(f"{w.inner.name()}_f64")
    vars = vars_from_shape(w.shape)
    f_f64[vars] = hl.cast(hl.Float(64), w.inner[vars])
    w_f64 = Wrapper(inner=f_f64, shape=w.shape)

    return sum(w_f64, axis=axis, schedule_strategy=schedule_strategy) / summed_element_count

def abs(w: Wrapper, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    f = hl.Func("abs")
    vars = vars_from_shape(w.shape)
    f[vars] = hl.abs(w.inner[vars])

    return Wrapper(shape=w.shape, inner=f)
