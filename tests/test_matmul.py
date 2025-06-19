import numpy as np
import numlide as nl


def test_matmul_small():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array(
        [
            [1, 2],
            [4, 5],
            [7, 8],
        ]
    )
    np.testing.assert_equal(a @ b, nl.wrap(a) @ b)


def test_matmul_small_large():
    a = np.random.randn(1, 768)
    b = np.random.randn(768, 50257)

    np_result = a @ b
    print(f"{np_result.shape=}")

    nl_result = nl.wrap(a, name="matmul_test_a") @ nl.wrap(b, name="matmul_test_b")
    print(f"{nl_result.shape=}")

    np.testing.assert_allclose(np_result, nl_result.to_numpy())
