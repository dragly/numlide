# This code is derived from NumPy, licensed under the BSD 3-Clause License.
# See third-party/licenses/numpy.txt for details.

from typing import Any, Optional, Tuple
import halide as hl
from collections.abc import Callable


from .schedule import ScheduleStrategy
from .wrapper import Wrapper, wrap
from .typing import ArrayLike
from .utils import var_from_index, vars_from_shape, tr
import numpy as np


def array_split(w, indices_or_sections, axis=0):
    if not isinstance(w, Wrapper):
        w = wrap(w)
    try:
        Ntotal = w.shape[axis]
    except AttributeError:
        Ntotal = len(w)
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError("number sections must be larger than 0.") from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = [0] + extras * [Neach_section + 1] + (Nsections - extras) * [Neach_section]
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()

    sub_arrays = []
    sary = swapaxes(w, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arrays.append(swapaxes(sary[st:end], axis, 0))

    return sub_arrays


def split(w: ArrayLike, indices_or_sections, axis=0) -> list[Wrapper]:
    if not isinstance(w, Wrapper):
        w = wrap(w)
    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = w.shape[axis]
        if N % sections:
            raise ValueError("array split does not result in an equal division") from None
    return array_split(w, indices_or_sections, axis)


def swapaxes(w: ArrayLike, axis1: int, axis2: int) -> Wrapper:
    if not isinstance(w, Wrapper):
        w = wrap(w)

    variables = vars_from_shape(w.shape)
    variables_swapped = list(variables)
    variables_swapped[axis1] = variables[axis2]
    variables_swapped[axis2] = variables[axis1]

    shape_swapped = list(w.shape)
    shape_swapped[axis1] = w.shape[axis2]
    shape_swapped[axis2] = w.shape[axis1]

    inner = hl.Func("swapped_axes")
    inner[variables] = w.inner[variables_swapped]

    return Wrapper(
        shape=tuple(shape_swapped),
        inner=inner,
    )
