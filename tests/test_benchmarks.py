import numpy as np
import numlide as nl


def _structured_light(m, images):
    patterns = images.shape[-1]
    minimum = m.min(images, axis=2)
    maximum = m.max(images, axis=2)
    threshold = (minimum + maximum) / 2
    binary_code = threshold[:, :, m.newaxis] < images
    decimal_code = m.sum(2 ** (patterns - m.arange(0, patterns) - 1)[m.newaxis, m.newaxis, :] * binary_code, axis=2)
    return decimal_code


def _create_patterns():
    rows = 768
    columns = 1024
    patterns = 8
    images = np.zeros((rows, columns, patterns))

    for i in range(patterns):
        divisions = 2 ** (i + 1)
        step = columns / divisions
        period = 2 * step
        images[:, :, i] = np.ones((rows, columns)) * ((np.indices((rows, columns))[1] % period) > step)

    return images


def test_structured_light():
    images = _create_patterns()
    images_wrapped = nl.wrap(images)
    np.testing.assert_allclose(images, images_wrapped.to_numpy())

    result_np = _structured_light(np, images)
    result_nl = _structured_light(nl, images_wrapped)

    np.testing.assert_allclose(result_nl.to_numpy(), result_np)


def test_structured_light_numlide(benchmark):
    images = _create_patterns()
    images_wrapped = nl.wrap(images)
    np.testing.assert_allclose(images, images_wrapped.to_numpy())

    benchmark(_structured_light, nl, images_wrapped)


def test_structured_light_numpy(benchmark):
    images = _create_patterns()

    benchmark(_structured_light, np, images)
