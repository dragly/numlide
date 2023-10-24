import numlide as nl
import numlide.linalg
from numlide.schedule import ScheduleStrategy
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


def test_mean_benchmark(benchmark):
    np.random.seed(42)
    image = np.random.randn(640, 480)

    result_np = np.mean(image) * image
    result_nl = nl.mean(image) * image
    result_nl.to_numpy()

    def calculate_nl():
        return result_nl.to_numpy()

    benchmark(calculate_nl)

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


def test_structured_light():
    rows = 768
    columns = 1024
    patterns = 8
    images = np.zeros((rows, columns, patterns))

    for i in range(patterns):
        divisions = 2 ** (i + 1)
        step = columns / divisions
        period = 2 * step
        images[:, :, i] = np.ones((rows, columns)) * ((np.indices((rows, columns))[1] % period) > step)

    images_wrapped = nl.wrap(images)
    np.testing.assert_allclose(images, images_wrapped.to_numpy())

    def structured_light(m, images):
        minimum = m.min(images, axis=2)
        maximum = m.max(images, axis=2)
        threshold = (minimum + maximum) / 2
        binary_code = threshold[:, :, m.newaxis] < images
        decimal_code = m.sum(2 ** (patterns - m.arange(0, patterns) - 1)[m.newaxis, m.newaxis, :] * binary_code, axis=2)
        return decimal_code

    result_np = structured_light(np, images)
    result_nl = structured_light(nl, images_wrapped)

    np.testing.assert_allclose(result_nl.to_numpy(), result_np)


if __name__ == "__main__":
    test_structured_light()
