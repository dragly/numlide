from __future__ import annotations
from collections.abc import Callable
from typing import Any, Sequence, Tuple
import halide as hl
import numpy as np
from dataclasses import dataclass
from . import linalg
from .math import sin, cos, tan, sqrt, sum, mean, min, max
from .wrapper import Wrapper, wrap, array
from .ranges import arange

newaxis = np.newaxis

