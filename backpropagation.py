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

def min_d(L, a, y, thetas): #1 capa
    l = L -1
    d = [0] * l
    d[-1] = y - a[-1]

    d[0] = d[1] @ thetas[1] * a[1] * (1 - a[1])
    return d

def may_d(L, a, y, thetas, D): #1 capa
    l = L - 1
    Dc = [0] * l

    d = min_d(L, a, y, thetas)
    Dc[0] = D[0] + a[0].T @ d[0]
    Dc[1] = D[1] + a[1].T @ d[1]

    Df = []
    for f in (Dc):
        Df.extend(f.ravel())
    D = np.asarray(Df)
    return D


