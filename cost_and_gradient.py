import numpy as np
from PIL import Image
import PIL.ImageOps

from backpropagation import may_d
import feed_forward


# theta1Size = (3, 2)
# theta1Len = theta1Size[0] * theta1Size[1]
# theta2Size = (1, 3)
# theta2Len = theta2Size[0] * theta2Size[1]
# thetasLen = theta1Len * theta2Len
#
# bias1Size = (3, 1)
# bias1Len = bias1Size[0] * bias1Size[1]
# bias2Size = (1, 1)
# bias2Len = bias2Size[0] * bias2Size[1]
# biasLen = bias1Len * bias2Len




def cost(h, y):
    j = (np.nan_to_num(y * np.log(h)) + np.nan_to_num((1 - y) * np.log(1 - h))).mean()
    # j = ((h - y) ** 2).mean() / 2
    return j


# def cost_and_gradient(x, y, theta, bias, Dw):
#     L = 3
#
#     theta1 = theta[:(theta1Size[0] * theta1Size[1])].reshape(theta1Size)
#     theta2 = theta[(theta1Size[0] * theta1Size[1]):].reshape(theta2Size)
#
#     bias1 = bias[0]
#     bias2 = bias[1]
#
#     r, a, weights, biases = feed_forward.feed_forward(x, theta1, theta2, bias1, bias2)
#
#     bw = may_d(L, a, y, weights, Dw)
#
#     w1 = bw[:(theta1Size[0] * theta1Size[1])].reshape(theta1Size)
#     w2 = bw[(theta1Size[0] * theta1Size[1]):].reshape(theta2Size)
#
#     b1 = np.sum(w1.T, axis=0, keepdims=True)
#     b2 = np.sum(w2.T, axis=0, keepdims=True)
#
#     bb = np.hstack((
#         b1.ravel(),
#         b2.ravel()
#     ))
#
#     theta_and_bias = np.hstack((
#         bw,
#         bb
#     ))
#



def cost_and_gradient_two(x, y, theta, bias, Dw):
    L = 5

    theta1 = theta[0]
    theta2 = theta[1]
    theta3 = theta[2]
    theta4 = theta[3]

    bias1 = bias[0]
    bias2 = bias[1]
    bias3 = bias[2]
    bias4 = bias[3]

    r, a, weights, biases = feed_forward.feed_forward_two(x, theta1, theta2, theta3, theta4, bias1, bias2, bias3, bias4)

    bw = may_d(L, a, y, weights, Dw)

    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    w4 = weights[3]

    b1 = np.sum((bw[0] / bw[0].max()), axis=0, keepdims=True)
    b2 = np.sum((bw[1] / bw[1].max()), axis=0, keepdims=True)
    b3 = np.sum((bw[2] / bw[2].max()), axis=0, keepdims=True)
    b4 = np.sum((bw[2] / bw[3].max()), axis=0, keepdims=True)

    bb = np.hstack((
        b1.ravel(),
        b2.ravel(),
        b3.ravel(),
        b4.ravel(),
    ))

    theta_and_bias = np.hstack((
        w1.ravel(),
        w2.ravel(),
        w3.ravel(),
        w4.ravel(),
        bb.ravel(),
    ))

    return cost(r.T, y), bw, bb, theta_and_bias

