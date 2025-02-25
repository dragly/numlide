import numpy as np
import numlide as nl


def test_slice():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    xw = nl.wrap(x)
    np.testing.assert_equal(x[1:7:2], xw[1:7:2])


def test_fewer_indices():
    x = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
    )
    xw = nl.wrap(x)

    np.testing.assert_equal(x[0], xw[0])
