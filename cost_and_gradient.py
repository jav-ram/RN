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


theta1Size = (16, 784)
theta1Len = theta1Size[0] * theta1Size[1]

theta2Size = (16, 16)
theta2Len = theta2Size[0] * theta2Size[1]

theta3Size = (6, 16)
theta3Len = theta3Size[0] * theta3Size[1]

thetasLen = theta1Len + theta2Len + theta3Len


bias1Size = (16, 1)
bias1Len = bias1Size[0] * bias1Size[1]

bias2Size = (16, 1)
bias2Len = bias2Size[0] * bias2Size[1]

bias3Size = (6, 1)
bias3Len = bias3Size[0] * bias3Size[1]

biasLen = bias1Len + bias2Len + bias3Len


def cost(h, y):
    # j = (np.nan_to_num(y * np.log(h)) + np.nan_to_num((1 - y) * np.log(1 - h))).mean()
    j = ((h - y) ** 2).mean() / 2
    return j


def cost_and_gradient(x, y, theta, bias, Dw):
    L = 3

    theta1 = theta[:(theta1Size[0] * theta1Size[1])].reshape(theta1Size)
    theta2 = theta[(theta1Size[0] * theta1Size[1]):].reshape(theta2Size)

    bias1 = bias[:(bias1Size[0] * bias1Size[1])].reshape(bias1Size)
    bias2 = bias[(bias1Size[0] * bias1Size[1]):].reshape(bias2Size)

    r, a, weights, biases = feed_forward.feed_forward(x, theta1, theta2, bias1, bias2)

    bw = may_d(L, a, y, weights, Dw)

    w1 = bw[:(theta1Size[0] * theta1Size[1])].reshape(theta1Size)
    w2 = bw[(theta1Size[0] * theta1Size[1]):].reshape(theta2Size)

    b1 = np.sum(w1.T, axis=0, keepdims=True)
    b2 = np.sum(w2.T, axis=0, keepdims=True)

    bb = np.hstack((
        b1.ravel(),
        b2.ravel()
    ))

    theta_and_bias = np.hstack((
        bw,
        bb
    ))




def cost_and_gradient_two(x, y, theta, bias, Dw):
    L = 4

    theta_array = np.split(theta, [theta1Len, theta1Len + theta2Len, theta1Len + theta2Len + theta3Len])
    theta1 = theta_array[0].reshape(theta1Size)
    theta2 = theta_array[1].reshape(theta2Size)
    theta3 = theta_array[2].reshape(theta3Size)

    bias_array = np.split(bias, [bias1Len, bias1Len + bias2Len, bias1Len + bias2Len + bias3Len])
    bias1 = bias_array[0].reshape(bias1Size)
    bias2 = bias_array[1].reshape(bias2Size)
    bias3 = bias_array[2].reshape(bias3Size)

    r, a, weights, biases = feed_forward.feed_forward_two(x, theta1, theta2, theta3, bias1, bias2, bias3)

    bw = may_d(L, a, y, weights, Dw)

    w_array = np.split(bw, [theta1Len, theta1Len + theta2Len, theta1Len + theta2Len + theta3Len])
    w1 = w_array[0].reshape(theta1Size)
    w2 = w_array[1].reshape(theta2Size)
    w3 = w_array[2].reshape(theta3Size)

    b1 = np.sum(w1.T, axis=0, keepdims=True)
    b2 = np.sum(w2.T, axis=0, keepdims=True)
    b3 = np.sum(w3.T, axis=0, keepdims=True)
    bb = np.hstack((
        b1.ravel(),
        b2.ravel(),
        b3.ravel()
    ))
    theta_and_bias = np.hstack((
        bw.ravel(),
        bb.ravel()
    ))

    return cost(r.T, y), bw, bb, theta_and_bias

