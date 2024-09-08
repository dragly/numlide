import numlide as nl
import numpy as np


def test_linalg_norm():
    data = np.array([[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11], [11, 12, 13, 14, 15, 16]])

    result_nl = nl.linalg.norm(data)
    result_np = np.linalg.norm(data)

    np.testing.assert_allclose(result_nl.to_numpy(), result_np)


def test_box_filter():
    image = np.random.randn(64, 64)

    def filtered(image):
        return image[1:-1, 1:-1] + image[2:, 1:-1] + image[:-2, 1:-1] + image[1:-1, 2:] + image[1:-1, :-2]

    result_np = filtered(image)
    result_nl = filtered(nl.wrap(image))

    np.testing.assert_allclose(result_nl.to_numpy(), result_np)


def test_math():
    np.random.seed(42)
    image = np.random.randn(640, 480)

    def compare(np_method, nl_method):
        result_np = np_method(image)
        result_nl = nl_method(image)

        np.testing.assert_allclose(result_nl.to_numpy(), result_np)

        for axis in [0, 1]:
            result_np = np_method(image, axis=axis)
            result_nl = nl_method(image, axis=axis)

            np.testing.assert_allclose(result_nl.to_numpy(), result_np)

    compare(np.min, nl.min)
    compare(np.max, nl.max)
    compare(np.mean, nl.mean)
    compare(np.sum, nl.sum)

def test_numpy():
    a = np.array([1, 2, 3])
    b = nl.array([1, 2, 3])
    np.testing.assert_equal(a + b, np.array([2, 4, 6]))
