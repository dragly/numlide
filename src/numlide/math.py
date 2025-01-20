from typing import Any, Optional, Tuple
import halide as hl
from collections.abc import Callable

from .schedule import ScheduleStrategy
from .wrapper import Wrapper, wrap
from .utils import var_from_index, vars_from_shape, tr
from .typing import ArrayLike
import numpy as np


def apply(w: Wrapper, f: Callable[[hl.Expr], hl.Expr]) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)
    func = hl.Func(f"apply_{f.__name__}")
    variables = vars_from_shape(w.shape)
    func[variables] = f(w.inner[variables])
    return Wrapper(inner=func, shape=w.shape)


def sin(w: ArrayLike) -> Wrapper:
    return apply(w, hl.sin)


def cos(w: ArrayLike) -> Wrapper:
    return apply(w, hl.cos)


def tan(w: ArrayLike) -> Wrapper:
    return apply(w, hl.tan)


def tanh(w: ArrayLike) -> Wrapper:
    return apply(w, hl.tanh)


def sqrt(w: ArrayLike) -> Wrapper:
    return apply(w, hl.sqrt)


def exp(w: ArrayLike) -> Wrapper:
    return apply(w, hl.exp)


def _wrap_axis(axis: int, ndim: int) -> int:
    if axis >= 0:
        return axis
    else:
        return ndim + axis


def _deduce_axis(
    wrapper: Wrapper,
    axis: Optional[int | Tuple[int, ...]],
) -> Tuple[int, ...]:
    if axis is None:
        return tuple(i for i in range(wrapper.ndim))
    elif isinstance(axis, int):
        return tuple((_wrap_axis(axis, ndim=wrapper.ndim),))
    return tuple(_wrap_axis(ax, ndim=wrapper.ndim) for ax in axis)


def _reduce(
    w: Wrapper,
    impl: Callable[[Any, Any], hl.Func],
    axis: Optional[int | Tuple[int, ...]],
    keepdims: bool,
    schedule_strategy=ScheduleStrategy.auto,
):
    deduced_axis = _deduce_axis(w, axis)

    rdom_elements = list()
    for i, extent in enumerate(w.shape):
        if i in deduced_axis:
            rdom_elements.append((0, extent))

    shape = tuple()
    next_rdom_element = 0
    rdom = hl.RDom(tr(rdom_elements))
    right_variables = []
    left_variables = []
    for i in tr(range(len(w.shape))):
        var = var_from_index(i)
        if i in deduced_axis:
            right_variables.append(rdom[next_rdom_element])
            next_rdom_element += 1
            if keepdims:
                left_variables.append(var)
                shape += (1,)
        else:
            right_variables.append(var)
            left_variables.append(var)
            shape += (w.shape[i],)

    shape = tr(shape)

    f = impl(left_variables, right_variables)

    if schedule_strategy == ScheduleStrategy.auto:
        f.compute_root()

    return Wrapper(inner=f, shape=shape)


def sum(
    w: Wrapper,
    keepdims: bool = False,
    axis: Optional[int | Tuple[int, ...]] = None,
    schedule_strategy=ScheduleStrategy.auto,
) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    def impl(left_variables, right_variables) -> hl.Func:
        f = hl.Func("sum")
        f[left_variables] = hl.cast(w.inner.type(), 0)
        f[left_variables] += w.inner[right_variables]
        return f

    return _reduce(w, impl=impl, axis=axis, keepdims=keepdims, schedule_strategy=schedule_strategy)


def min(
    w: Wrapper, keepdims: bool = False, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto
) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    def impl(left_variables, right_variables) -> hl.Func:
        f = hl.Func("min")
        f[left_variables] = hl.minimum(w.inner[right_variables])
        return f

    return _reduce(w, impl=impl, keepdims=keepdims, axis=axis, schedule_strategy=schedule_strategy)


def max(
    w: Wrapper, keepdims: bool = False, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto
) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    def impl(left_variables, right_variables) -> hl.Func:
        f = hl.Func("max")
        f[left_variables] = hl.maximum(w.inner[right_variables])
        return f

    return _reduce(w, impl=impl, axis=axis, keepdims=keepdims, schedule_strategy=schedule_strategy)


def argmax(
    w: Wrapper, keepdims: bool = False, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto
) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    def impl(left_variables, right_variables) -> hl.Func:
        f = hl.Func("argmax")
        f[left_variables] = hl.argmax(w.inner[right_variables])[0]
        return f

    if axis is None:
        w = w.flatten()

    return _reduce(w, impl=impl, axis=axis, keepdims=keepdims, schedule_strategy=schedule_strategy)


def mean(
    w: Wrapper,
    axis: Optional[int | Tuple[int, ...]] = None,
    schedule_strategy=ScheduleStrategy.auto,
    keepdims: bool = False,
) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    deduced_axis = _deduce_axis(w, axis)
    summed_element_count = np.prod(np.array(w.shape)[list(deduced_axis)])

    f_f64 = hl.Func(f"{w.inner.name()}_f64")
    vars = vars_from_shape(w.shape)
    f_f64[vars] = hl.cast(hl.Float(64), w.inner[vars])
    w_f64 = Wrapper(inner=f_f64, shape=w.shape)

    return sum(w_f64, axis=deduced_axis, schedule_strategy=schedule_strategy, keepdims=keepdims) / summed_element_count


def var(
    w: Wrapper, axis: Optional[int | Tuple[int]] = None, schedule_strategy=ScheduleStrategy.auto, keepdims: bool = False
) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    deduced_axis = _deduce_axis(w, axis)
    summed_element_count = np.prod(np.array(w.shape)[list(deduced_axis)])

    mean_value = mean(w, axis=axis, schedule_strategy=schedule_strategy, keepdims=True)
    squared = (w - mean_value) ** 2

    return sum(squared, axis=deduced_axis, keepdims=keepdims) / summed_element_count


def abs(w: Wrapper) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    f = hl.Func("abs")
    vars = vars_from_shape(w.shape)
    f[vars] = hl.abs(w.inner[vars])

    return Wrapper(shape=w.shape, inner=f)
