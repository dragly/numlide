import numlide as nl
import numlide.linalg
import numpy as np

data = np.array([[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11], [11, 12, 13, 14, 15, 16]])
data2 = np.array([[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11], [11, 12, 13, 14, 15, 16]])

datal = nl.array(data)
datal2 = nl.array(data2)

def impl(n, a, b):
    return n.linalg.norm(a)

result_nl = impl(nl, datal, datal2)
result_np = impl(np, data, data2)

np.testing.assert_allclose(result_nl.to_numpy(), result_np)
