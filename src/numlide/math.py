import halide as hl
from collections.abc import Callable

from .schedule import ScheduleStrategy
from .wrapper import Wrapper, wrap
from .utils import vars_from_shape, tr
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


def sum(w: Wrapper, schedule_strategy=ScheduleStrategy.auto) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    f = hl.Func("sum")
    f[()] = hl.cast(w.inner.type(), 0)
    rdom_elements = list()
    for extent in w.shape:
        rdom_elements.append((0, extent))

    rdom = hl.RDom(tr(rdom_elements))
    rdom_accesors = []
    for i in range(rdom.dimensions()):
        rdom_accesors.append(rdom[i])
    f[()] += w.inner[rdom_accesors]

    if schedule_strategy == ScheduleStrategy.auto:
        f.compute_root()

    return Wrapper(inner=f, shape=tuple())


def mean(w: Wrapper, schedule_strategy=ScheduleStrategy.auto) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    return sum(w, schedule_strategy=ScheduleStrategy.auto) / np.prod(w.shape)
