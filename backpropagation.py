import numpy as np
import math

Dc = []

# def min_d(L, a, y, thetas):
#     l = L - 1
#     d = [None] * l
#     d[-1] = a[-1] - y
#     for i in reversed(range(l-1)):
#         print(d[i+1].shape, thetas[i+1].shape, a[i+1].shape)
#         d[i] = (d[i+1] @ thetas[i+1]) * a[i+1] * (1 - a[i+1])
#     print(d[0].shape, d[1].shape)
#     return d

def min_d(L, a, y, thetas):
    l = L -1
    d = [0] * l
    d[-1] = y - a[-1]

    for i in reversed(range(l-1)):
        d[i] = d[i+1] @ thetas[i+1] * a[i+1] * (1 - a[i+1])
    return d

def may_d(L, a, y, thetas, D):
    l = L - 1
    Dc = [0] * l

    d = min_d(L, a, y, thetas)
    for i in range(l):
        Dc[i] = D[i] + a[i].T @ d[i]

    Df = []
    for f in range(len(Dc)):
        Df.extend(list(Dc[f].ravel()))
    D = np.asarray(Df)
    return D

