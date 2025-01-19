import numpy as np
import numlide as nl


def test_swapaxes():
    x = np.array([[1, 2, 3]])
    np.testing.assert_equal(np.swapaxes(x, 0, 1), nl.swapaxes(x, 0, 1))

    x = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    np.testing.assert_equal(np.swapaxes(x, 0, 2), nl.swapaxes(x, 0, 2))


def test_split():
    x = np.arange(9.0)
    np.testing.assert_equal(np.split(x, 3), nl.split(x, 3))


def test_array_split():
    x = np.arange(8.0)
    np.testing.assert_equal(np.array_split(x, 3), nl.array_split(x, 3))

    x = np.arange(9)
    np.testing.assert_equal(np.array_split(x, 4), nl.array_split(x, 4))


def test_concatenate():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    np.testing.assert_equal(
        np.concatenate((a, b), axis=0),
        nl.concatenate(
            (a, b),
            axis=0,
        ),
    )
    np.testing.assert_equal(
        np.concatenate((a, b.T), axis=1),
        nl.concatenate(
            (a, b.T),
            axis=1,
        ),
    )
