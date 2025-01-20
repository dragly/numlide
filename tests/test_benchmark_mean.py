import numpy as np
import numlide as nl


def _create_image():
    return np.random.randn(1920, 2560)


def test_mean_numpy(benchmark):
    image = _create_image()

    benchmark(np.mean, image)


def test_mean_numlide(benchmark):
    image = _create_image()
    image_wrapped = nl.wrap(image)
    np.testing.assert_allclose(image, image_wrapped.to_numpy())

    benchmark(nl.mean, image)
