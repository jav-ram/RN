import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feed_forward(X, theta1, theta2):
    m, n = X.shape
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)

    m, n = a2.shape
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    thetas, a = [], []
    a.extend((a1, a2, a3))
    thetas.extend((theta1, theta2))

    return a3, a, thetas


def feed_forward_two(X, theta1, theta2, theta3):
    m, n = X.shape
    a1 = np.hstack((
        np.ones(m).reshape((m, 1)),
        X
    ))
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)

    m, n = a2.shape
    a2 = np.hstack((
        np.ones(m).reshape((m, 1)),
        a2
    ))
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    m, n = a3.shape
    a3 = np.hstack((
        np.ones(m).reshape((m, 1)),
        a3
    ))
    z4 = a3 @ theta3.T
    a4 = sigmoid(z4)

    return a4


def feed_forward_three(X, theta1, theta2, theta3, theta4):
    m, n = X.shape
    a1 = np.hstack((
        np.ones(m).reshape((m, 1)),
        X
    ))
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)

    m, n = a2.shape
    a2 = np.hstack((
        np.ones(m).reshape((m, 1)),
        a2
    ))
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)

    m, n = a3.shape
    a3 = np.hstack((
        np.ones(m).reshape((m, 1)),
        a3
    ))
    z4 = a3 @ theta3.T
    a4 = sigmoid(z4)

    m, n = a4.shape
    a4 = np.hstack((
        np.ones(m).reshape((m, 1)),
        a4
    ))
    z5 = a4 @ theta4.T
    a5 = sigmoid(z5)

    return a5


x = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])
t1 = np.array([
    [0.5, 0.1],
    [0.3, 0.1]
])
t2 = np.array([[0.7, 0.8, 0.9]])
y = np.array([
    [2],
    [3],
    [4],
    [5],
    [6],
    [7]
])
#r, a, t = feed_forward(x, t1, t2)

x = np.array([[1, 2], [2, 4], [2, 4]])
t1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.7, 0.8, 0.9]])
t2 = np.array([[0.7, 0.8, 0.9, 0.6], [0.3, 0.2, 0.5, 0.7]])
t3 = np.array([[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]])
a = feed_forward_two(x, t1, t2, t3)
