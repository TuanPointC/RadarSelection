import numpy as np
denta_t = 0.1
E_v = np.array([denta_t ** 3 / 3, 0, denta_t ** 2 / 2, 0, 0, denta_t ** 3 / 3, 0, denta_t ** 2 / 2, denta_t / 2,
                0, denta_t, 0, 0, denta_t / 2, 0, denta_t]).reshape(4, 4)
# print(E_v.shape)
# print(np.std(np.array([1, 2, 5, 7]).reshape(2, 2), axis=1))
print(np.random.normal(0, np.std(E_v, axis=1)).reshape(4, 1))

