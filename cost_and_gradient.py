import numpy as np
import math

from backpropagation import may_d
import feed_forward


theta1Size = (2, 2)
theta2Size = (2, 2)


def cost(h, y):
    j = (y * np.log(h) + np.nan_to_num((1 - y) * np.log(1 - h))).mean()
    return j



def cost_and_gradient(x, y, theta, D):
    L = 3
    theta1 = theta[:(theta1Size[0] * theta1Size[1])].reshape(theta1Size)
    theta2 = theta[(theta1Size[0] * theta1Size[1]):].reshape(theta2Size)

    r, a, thetas = feed_forward.feed_forward(x, theta1, theta2)
    return cost(r, y), may_d(L, a, y, thetas, D)
