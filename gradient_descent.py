import numpy as np
import math


from cost_and_gradient import cost_and_gradient
from feed_forward import feed_forward

norm = lambda v: ((v ** 2.0).sum()) ** 0.5

def gradient_descent(
        X,
        y,
        theta_0,
        cost_and_gradient,
        alpha=1,
        threshold=0.001,
        max_iter=10000):
    theta, last_cost, i = theta_0, 999999999999, 0
    D = [0.0] * 8
    while i < max_iter and norm(cost_and_gradient(X, y, theta, D)[1]) > threshold:
        cost, gradient = cost_and_gradient(X, y, theta, D)
        theta -= alpha * gradient
        D = gradient
        i += 1


    return theta

pair = lambda x: x % 2

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
t1 = np.array([
    [1.12675663e+08, 1.12668322e+08], [1.12666336e+08, 1.12659518e+08]
])
t2 = np.array([
    [2.11247613e+09, 1.61254707e+09], [2.11228072e+09, 1.61254766e+09]
])
y = np.array([
    [1,0],
    [0,1],
    [1,0],
    [0,1]
])

# x = np.array([[1.0], [2.0], [3.0]])
# t1 = np.array([[1.0], [0.0]])
# t2 = np.array([[0.0, 2.0]])
# y = np.array([[0.0], [0.0], [0.0]])

# r, t, g = feed_forward(x, t1, t2)
#
# print(r)

print(gradient_descent(
        x,
        y,
        np.hstack((t1.ravel(), t2.ravel())),
        cost_and_gradient,
))
