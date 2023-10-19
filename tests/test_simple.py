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
        return (
            image[1:-1, 1:-1] + image[2:, 1:-1] + image[:-2, 1:-1] + image[1:-1, 2:] + image[1:-1, :-2]
        )

    result_np = filtered(image)
    result_nl = filtered(nl.wrap(image))

    np.testing.assert_allclose(result_nl.to_numpy(), result_np)


def test_mean():
    np.random.seed(42)
    image = np.random.randn(64, 48)

    result_np = np.mean(image)
    result_nl = nl.mean(image)

    out = result_nl * image

    np.testing.assert_allclose(result_nl.to_numpy(), result_np)


if __name__ == "__main__":
    test_mean()
