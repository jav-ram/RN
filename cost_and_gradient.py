import numpy as np
import math

from backpropagation import may_d
import feed_forward


theta1Size = (2, 2)
theta2Size = (1, 2)


def cost(h, y):
    m, n = h.shape
    j = -(1/m) * (y * np.log(h) + (1 - y) * np.log(1 - h)).sum()
    return j


H = np.array([[0.7], [0.9999], [0.000000000001]])
Y = np.array([[1], [1], [0]])



def cost_and_gradient(x, y, theta, D):
    L = 3
    theta1 = theta[:(theta1Size[0] * theta1Size[1])].reshape(theta1Size)
    theta2 = theta[(theta1Size[0] * theta1Size[1]):].reshape(theta2Size)

    r, a, thetas = feed_forward.feed_forward(x, theta1, theta2)
    return cost(x, y), may_d(L, a, y, thetas, D)
