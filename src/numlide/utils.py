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


def calculate_extent(start, stop, step):
    return int(np.ceil(np.abs(stop - start) / np.max([np.abs(step), 1])))
