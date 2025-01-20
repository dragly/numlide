import numpy as np
import numlide as nl


def test_set_single():
    a = np.array([[1, 2], [3, 4]])
    aw = nl.wrap(a)

    a[0, 0] = 5
    aw[0, 0] = 5

    np.testing.assert_equal(a, aw.to_numpy())


def test_set_slice_to_single():
    a = np.array([[1, 2], [3, 4]])
    aw = nl.wrap(a)

    a[:, 0] = 5
    aw[:, 0] = 5

    np.testing.assert_equal(a, aw.to_numpy())
