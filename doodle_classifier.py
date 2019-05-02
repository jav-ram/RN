import numpy as np
from PIL import Image
import PIL.ImageOps

import random

from feed_forward import feed_forward_two
from gradient_descent import gradient_descent
from cost_and_gradient import cost_and_gradient_two

# Load Train set
X = np.load('./train/X.npy')
Y = np.load('./train/Y.npy')

theta1Size = (14, 784)
theta1Len = theta1Size[0] * theta1Size[1]

theta2Size = (10, 14)
theta2Len = theta2Size[0] * theta2Size[1]

theta3Size = (6, 10)
theta3Len = theta3Size[0] * theta3Size[1]

thetasLen = theta1Len + theta2Len + theta3Len

# Thetas
t1 = np.random.uniform(low=0, high=1, size=theta1Len).reshape(theta1Size)
t2 = np.random.uniform(low=0, high=1, size=theta2Len).reshape(theta2Size)
t3 = np.random.uniform(low=0, high=1, size=theta3Len).reshape(theta3Size)
# Bias
b1 = random.random()
b2 = random.random()

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
    (t1, t2, t3),
    (b1, b2),
    cost_and_gradient_two,
    alpha=0.0001,
    beta=0.05,
    threshold=2.71,
    max_iter=10000
)

np.save('dw', theta)
np.save('db', bias)

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

