import halide as hl
from collections.abc import Callable
from .wrapper import Wrapper, array
from .utils import vars_from_shape, tr


def apply(w: Wrapper, f: Callable[[hl.Expr], hl.Expr]) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = array(w)
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
    if not isinstance(w, Wrapper):
        w = array(w)

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
