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


def vars_from_shape(shape: Tuple[int], zero_if_one: bool = False) -> Tuple[hl.Var, ...]:
    variables = tuple()
    for i in range(len(shape)):
        if zero_if_one and shape[i] == 1:
            variables += (0,)
        else:
            variables += (var_from_index(i),)
    return tr(variables)


def calculate_extent(start, stop, step):
    return int(np.ceil(np.abs(stop - start) / np.max([np.abs(step), 1])))


def halide_type(dtype: Any) -> hl.Type:
    match dtype:
        case np.float32:
            return hl.Float(32)
        case np.float64:
            return hl.Float(64)
        case np.int8:
            return hl.Int(8)
        case np.int16:
            return hl.Int(16)
        case np.int32:
            return hl.Int(32)
        case np.int64:
            return hl.Int(64)
        case np.uint8:
            return hl.UInt(8)
        case np.uint16:
            return hl.UInt(16)
        case np.uint32:
            return hl.UInt(32)
        case np.uint64:
            return hl.UInt(64)
        case np.bool_:
            return hl.Bool()
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")
