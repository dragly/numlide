import numlide as nl
import halide as hl


def norm(w: nl.Wrapper):
    return nl.sqrt(nl.sum(w**2))
