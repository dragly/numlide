import numpy as np
import numlide as nl


def test_generators():
    np.testing.assert_equal(
        nl.ones((2, 3)).to_numpy(),
        np.ones((2, 3)),
    )
    np.testing.assert_equal(
        nl.zeros((2, 3)).to_numpy(),
        np.zeros((2, 3)),
    )
    np.testing.assert_equal(
        nl.ones((2, 3), dtype=np.int16).to_numpy(),
        np.ones((2, 3), dtype=np.int16),
    )
    np.testing.assert_equal(
        nl.zeros((2, 3), dtype=np.int16).to_numpy(),
        np.zeros((2, 3), dtype=np.int16),
    )
