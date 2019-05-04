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

def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
vrelu_prime = np.vectorize(relu_prime)

def min_d(L, a, y, thetas):  # 4 layer
    l = L
    d = [0] * l

    d[4] = y.T - a[4]

    d[3] = (np.matmul(thetas[3].T, d[4]) * a[3] * (1 - a[3]))
    d[2] = (np.matmul(thetas[2].T, d[3]) * relu_prime(a[2]))
    d[1] = (np.matmul(thetas[1].T, d[2]) * relu_prime(a[1]))
    d[0] = (np.matmul(thetas[0].T, d[1]) * relu_prime(a[0]))

    return (d[0], d[1], d[2], d[3], d[4])


def may_d(L, a, y, thetas, D):  # 4 layer
    l = L - 1
    Dc = [0] * l

    d = min_d(L, a, y, thetas)

    Dc[0] = (D[0] + np.matmul(d[1], a[0].T) + 1000 * thetas[0]) / 7000
    Dc[1] = (D[1] + np.matmul(d[2], a[1].T) + 1000 * thetas[1]) / 7000
    Dc[2] = (D[2] + np.matmul(d[3], a[2].T) + 1000 * thetas[2]) / 7000
    Dc[3] = (D[3] + np.matmul(d[4], a[3].T) + 1000 * thetas[3]) / 7000

    return (Dc[0], Dc[1], Dc[2], Dc[3])


