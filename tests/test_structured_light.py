import numlide as nl
import numlide.linalg
from numlide.schedule import ScheduleStrategy
import numpy as np

def test_structured_light():
    np.random.seed(42)
    images = np.random.randn(640, 480, 10)

    def structured_light(m, images):
        minimum = m.min(images, axis=2)
        maximum = m.max(images, axis=2)
        threshold = (maximum + minimum) / 2
        binary_code = images > threshold[:, :, np.newaxis]
        print(binary_code.shape)
        decimal_code = binary_code * (2 ** np.arange(0, 10))
        print(decimal_code.shape)
        return decimal_code

    result_np = structured_light(np, images)
    result_nl = structured_light(nl, images)

    np.testing.assert_allclose(result_nl.to_numpy(), result_np)
