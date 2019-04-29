import numpy as np
import math

Dc = []

# def min_d(L, a, y, thetas): # 2 layer
#     l = L - 1
#     d = [None] * l
#     d[-1] = a[-1] - y
#     for i in reversed(range(l-1)):
#         print(d[i+1].shape, thetas[i+1].shape, a[i+1].shape)
#         d[i] = (d[i+1] @ thetas[i+1]) * a[i+1] * (1 - a[i+1])
#     print(d[0].shape, d[1].shape)
#     return d


def min_d(L, a, y, thetas):  # 4 layer
    l = L - 1
    d = [0] * l

    d[2] = a[3] * y.T

    d[1] = (thetas[2].T @ d[2]) * a[2] * (1 - a[2])
    d[0] = (thetas[1].T @ d[1]) * a[1] * (1 - a[1])

    return d


def may_d(L, a, y, thetas, D):  # 4 layer
    l = L - 1
    Dc = [0] * l

    d = min_d(L, a, y, thetas)

    Dc[0] = (D[0] + d[0] @ a[0].T)
    Dc[1] = (D[1] + d[1] @ a[1].T + thetas[1])
    Dc[2] = (D[2] + d[2] @ a[2].T + thetas[2])

    D = np.hstack((
        Dc[0].T.ravel(),
        Dc[1].T.ravel(),
        Dc[2].T.ravel(),
    ))

    return np.flip(D, 0).T * -1


