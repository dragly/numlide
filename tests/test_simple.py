import numlide as nl
import halide as hl
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
    values = np.array([[1.2, 3.4, 5.6], [0.1, 0.2, 0.3]])

    def compare(np_method, nl_method):
        np.testing.assert_allclose(
            np_method(values),
            nl_method(values).to_numpy(),
        )
        np.testing.assert_allclose(
            np_method(values),
            np_method(nl.wrap(values)),
        )

    compare(np.sqrt, nl.sqrt)
    compare(np.exp, nl.exp)
    compare(np.cos, nl.cos)
    compare(np.sin, nl.sin)
    compare(np.tan, nl.tan)
    compare(np.tanh, nl.tanh)


def test_matmul():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array(
        [
            [1, 2],
            [4, 5],
            [7, 8],
        ]
    )
    np.testing.assert_equal(a @ b, nl.wrap(a) @ b)


def test_numpy():
    a = np.array([1, 2, 4])
    b = nl.array([1, 2, 4])
    np.testing.assert_equal(a + b, np.array([2, 4, 8]))
    np.testing.assert_almost_equal(np.mean(a), np.mean(b))


def test_newaxis():
    def add_axis(array):
        return array[:, np.newaxis]

    array = np.array([1, 2, 3])
    result_np = add_axis(array)
    result_nl = add_axis(nl.wrap(array))

    np.testing.assert_equal(result_np, result_nl)


def test_broadcast():
    def multiply(array):
        return array * np.arange(0, 4)

    array = np.array(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            [
                [2, 3, 4, 5],
                [6, 7, 8, 9],
            ],
        ]
    )

    np.testing.assert_equal(
        multiply(array),
        multiply(nl.array(array)),
    )
