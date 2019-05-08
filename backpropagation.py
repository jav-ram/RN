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

    d[3] = a[3] - y.T

    # d[3] = (np.dot(thetas[3].T, d[4]) * a[3] * (1 - a[3]))
    d[2] = -(np.dot(thetas[2].T, d[3]) * a[2] * (1 - a[2]))
    d[1] = -(np.dot(thetas[1].T, d[2]) * a[1] * (1 - a[1]))
    d[0] = -(np.dot(thetas[0].T, d[1]) * a[0] * (1 - a[0]))

    return (d[0], d[1], d[2], d[3])


def may_d(L, a, y, thetas, D):  # 4 layer
    l = L - 1
    Dc = [0] * l

    d = min_d(L, a, y, thetas)

    Dc[0] = -(Dc[0] + np.dot(d[1], a[0].T)) / 7000
    Dc[1] = -(Dc[1] + np.dot(d[2], a[1].T)) / 7000
    Dc[2] = -(Dc[2] + np.dot(d[3], a[2].T)) / 7000

    return (Dc[0], Dc[1], Dc[2])



