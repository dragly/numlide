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
        binary_code = images > threshold
        decimal_code = binary_code * 2 ** np.arange(0, 10)

    result_np = structured_light(np, images)
    print(result_np[500, 500])

    # np.testing.assert_allclose(result_nl.to_numpy(), result_np)
