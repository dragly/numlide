import numpy as np
from numpy.testing import assert_equal
import numlide as nl


def test_multiply():
    a = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [8, 7, 6],
            ],
            [
                [5, 4, 3],
                [2, 1, 0],
            ],
        ],
    )
    b = np.array(
        [
            [
                [1, 2, 3],
            ],
        ],
    )
    assert_equal(a * b, nl.array(a) * nl.array(b))
