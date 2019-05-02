import numpy as np
from PIL import Image
import PIL.ImageOps

import random

from feed_forward import feed_forward_two
from gradient_descent import gradient_descent
from cost_and_gradient import cost_and_gradient_two

# Load Train set
X = np.load('./train/Xv1.npy')
Y = np.load('./train/Yv1.npy')

theta1Size = (16, 784)
theta1Len = theta1Size[0] * theta1Size[1]

theta2Size = (16, 16)
theta2Len = theta2Size[0] * theta2Size[1]

theta3Size = (10, 16)
theta3Len = theta3Size[0] * theta3Size[1]

theta4Size = (10, 10)
theta4Len = theta4Size[0] * theta4Size[1]

thetasLen = theta1Len + theta2Len + theta3Len + theta4Len

# Thetas
t1 = np.random.uniform(low=0, high=1, size=theta1Len).reshape(theta1Size)
t2 = np.random.uniform(low=0, high=1, size=theta2Len).reshape(theta2Size)
t3 = np.random.uniform(low=0, high=1, size=theta3Len).reshape(theta3Size)
t4 = np.random.uniform(low=0, high=1, size=theta4Len).reshape(theta4Size)
# Bias
b1 = np.ones(16).reshape((16, 1))
b2 = np.ones(16).reshape((16, 1))
b3 = np.ones(10).reshape((10, 1))
b4 = np.ones(10).reshape((10, 1))

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
    (t1, t2, t3, t4),
    (b1, b2, b3, b4),
    cost_and_gradient_two,
    alpha=0.1,
    beta=0.00000001,
    threshold=0.1,
    max_iter=10000
)

np.save('dwv1', theta)
np.save('dbv1', bias)

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

