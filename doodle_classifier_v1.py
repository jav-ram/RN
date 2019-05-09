import numpy as np
from PIL import Image
import PIL.ImageOps

import random

from feed_forward import feed_forward_two
from gradient_descent import gradient_descent
from cost_and_gradient import cost_and_gradient_two

# Load Train set
X = np.load('./train/train/x.npy')
Y = np.load('./train/train/y.npy')

# theta = np.load('theta.npy')
# bias = np.load('bias.npy')

theta1Size = (24, 784)
theta1Len = theta1Size[0] * theta1Size[1]

theta2Size = (20, 24)
theta2Len = theta2Size[0] * theta2Size[1]

theta3Size = (10, 20)
theta3Len = theta3Size[0] * theta3Size[1]

# theta4Size = (10, 16)
# theta4Len = theta4Size[0] * theta4Size[1]

thetasLen = theta1Len + theta2Len + theta3Len  # + theta4Len

# Thetas
theta1 = (np.random.uniform(low=0, high=1, size=theta1Len).reshape(theta1Size) * 2) - 1
theta2 = (np.random.uniform(low=0, high=1, size=theta2Len).reshape(theta2Size) * 2) - 1
theta3 = (np.random.uniform(low=0, high=1, size=theta3Len).reshape(theta3Size) * 2) - 1
# theta4 = (np.random.uniform(low=0, high=1, size=theta4Len).reshape(theta4Size) * 2) - 1
# Bias
bias1 = np.random.uniform(low=0, high=1, size=24).reshape((24, 1))
bias2 = np.random.uniform(low=0, high=1, size=20).reshape((20, 1))
bias3 = np.random.uniform(low=0, high=1, size=10).reshape((10, 1))
# bias4 = np.random.uniform(low=0, high=1, size=10).reshape((10, 1))

# theta = np.load('dw.npy')
# bias = np.load('db.npy')


# theta_array = np.split(theta, [theta1Len, theta1Len + theta2Len, theta1Len + theta2Len + theta3Len])
# t1 = theta_array[0].reshape(theta1Size)
# t2 = theta_array[1].reshape(theta2Size)
# t3 = theta_array[2].reshape(theta3Size)

# b1 = random.random()
# b2 = random.random()
# b3 = random.random()


theta, bias = gradient_descent(
    X,
    Y,
    (theta1, theta2, theta3),
    (bias1, bias2, bias3),
    cost_and_gradient_two,
    alpha=0.1,
    threshold=0.0001,
    max_iter=300
)


np.save('theta1', theta)
np.save('bias1', bias)

# theta = np.load('dw.npy')
# bias = np.load('db.npy')
#
#
# theta_array = np.split(theta, [theta1Len, theta1Len + theta2Len, theta1Len + theta2Len + theta3Len])
# t1 = theta_array[0].reshape(theta1Size)
# t2 = theta_array[1].reshape(theta2Size)
# t3 = theta_array[2].reshape(theta3Size)
#
# b1 = bias[0]
# b2 = bias[1]
#
# v = np.array(PIL.Image.open('./out/Square/2.jpg').convert("L")).ravel().reshape((1, 784)).astype(np.float) / 255
# v1 = np.array(PIL.Image.open('./out/Circle/2.jpg').convert("L")).ravel().reshape((1, 784)).astype(np.float) / 255
#
# print(feed_forward_two(
#     v.reshape(1, 784),
#     t1,
#     t2,
#     t3,
#     b1,
#     b2,
# )[0].T)
#
# print(feed_forward_two(
#     X[100].reshape(1, 784),
#     t1,
#     t2,
#     t3,
#     b1,
#     b2,
# )[0].T)

