import numpy as np
import numlide as nl


def _create_image():
    # return np.random.randn(16, 64)
    return np.random.randn(1920, 2560)
    # return np.random.randn(1920, 1920)
    # return np.random.randn(992, 992)


def test_matmul_numpy(benchmark):
    image = _create_image()

    def matmul():
        return image @ image.T

    benchmark(matmul)


def test_matmul_numlide(benchmark):
    image = _create_image()
    image_wrapped = nl.wrap(image)
    np.testing.assert_allclose(image, image_wrapped.to_numpy())

    def matmul():
        return (image_wrapped @ image_wrapped.T).to_numpy()

    benchmark(matmul)


def test_matmul_numlide_jit(benchmark):
    image = _create_image()
    image_wrapped = nl.wrap(image)
    np.testing.assert_allclose(image, image_wrapped.to_numpy())

    result = image_wrapped @ image_wrapped.T
    result.inner.compile_jit()

    def matmul():
        return result.to_numpy()

    benchmark(matmul)

    np.testing.assert_allclose(result.to_numpy(), image @ image.T)


# def test_matmul():
#     image = _create_image()
#     image_wrapped = nl.wrap(image)
#     result = image_wrapped @ image_wrapped.T
#     np.testing.assert_allclose(result.to_numpy(), image @ image.T)
