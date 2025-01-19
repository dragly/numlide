import numpy as np
import numlide as nl


def test_transpose_2d():
    a = np.array([[1, 2], [3, 4]])
    aw = nl.wrap(a)
    np.testing.assert_equal(a.transpose(), aw.transpose())


def test_transpose_1d():
    a = np.array([1, 2, 3, 4])
    aw = nl.wrap(a)
    np.testing.assert_equal(a.transpose(), aw.transpose())


def test_transpose_3d():
    a = np.ones((1, 2, 3))
    aw = nl.wrap(a)
    np.testing.assert_equal(a.transpose(), aw.transpose())


def test_transpose_4d():
    a = np.ones((2, 3, 4, 5))
    aw = nl.wrap(a)
    np.testing.assert_equal(a.transpose(), aw.transpose())


def test_transpose_custom():
    a = np.ones((2, 3, 4, 5))
    aw = nl.wrap(a)
    np.testing.assert_equal(
        a.transpose([3, 1, 2, 0]),
        aw.transpose([3, 1, 2, 0]),
    )
