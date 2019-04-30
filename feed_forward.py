import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.log(1. + np.exp(x))


def relu(x):
    return x * (x > 0)


def guassian(x):
    return np.exp((-1) * x**2)


def alpha(x):
    if x >= 0.96:
        return 1
    else:
        return 0


valpha = np.vectorize(alpha)

vsigmoid = np.vectorize(sigmoid)


def feed_forward(X, theta1, theta2, bias1, bias2):
    a1 = X.T

    z2 = theta1 @ a1
    a2 = sigmoid(z2 + bias1)

    z3 = theta2 @ a2
    a3 = sigmoid(z3 + bias2)

    thetas, a, biases = [], [], []
    a.extend((a1, a2, a3))
    thetas.extend((theta1, theta2))
    biases.extend((bias1, bias2))

    return a3, a, thetas, biases


def feed_forward_two(X, theta1, theta2, theta3, bias1, bias2, debug=False):
    a1 = X.T

    # z2 = np.matmul(theta1, a1)
    # a2 = vsigmoid((z2 - z2.min()) / 50000)
    #
    # z3 = np.matmul(theta2, a2)
    # a3 = vsigmoid(z3 - z3.min() + bias1)
    #
    # z4 = np.matmul(theta3, a3)
    # a4 = vsigmoid(z4 - z4.min() + bias2)

    z2 = np.matmul(theta1, a1)
    a2 = np.tanh(z2 / 500000)

    z3 = np.matmul(theta2, a2)
    a3 = np.tanh(z3 + bias1)

    z4 = np.matmul(theta3, a3)
    a4 = sigmoid(z4 + bias2)

    a = (a1, a2, a3, a4)
    thetas = (theta1, theta2, theta3)
    biases = (bias1, bias2)

    return a4, a, thetas, biases


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


# x = np.array([
#     [1],
#     [2],
#     [3],
#     [4],
#     [5],
#     [6]
# ])
# t1 = np.array([
#     [0.5, 0.1],
#     [0.3, 0.1]
# ])
# t2 = np.array([[0.7, 0.8, 0.9]])
# y = np.array([
#     [2],
#     [3],
#     [4],
#     [5],
#     [6],
#     [7]
# ])
# #r, a, t = feed_forward(x, t1, t2)
#
# x = np.array([[1, 2], [2, 4], [2, 4]])
# t1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.7, 0.8, 0.9]])
# t2 = np.array([[0.7, 0.8, 0.9, 0.6], [0.3, 0.2, 0.5, 0.7]])
# t3 = np.array([[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]])
# a = feed_forward_two(x, t1, t2, t3)
