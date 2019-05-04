import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


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
vrelu = np.vectorize(relu)


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


def feed_forward_two(X, theta1, theta2, theta3, theta4, bias1, bias2, bias3, bias4):
    a1 = (X.T - X.T.std()) / X.mean()

    z2 = theta1 @ a1 + bias1

    a2 = vrelu((z2 - z2.std()) / z2.mean())

    z3 = theta2 @ a2 + bias2
    a3 = vrelu((z3 - z3.std()) / z3.mean())

    z4 = theta3 @ a3 + bias3
    a4 = vrelu((z4 - z4.std()) / z4.mean())

    z5 = theta4 @ a4 + bias4
    a5 = vsigmoid((z5 - z5.std()) / z5.mean())

    a = (a1, a2, a3, a4, a5)
    thetas = (theta1, theta2, theta3, theta4)
    biases = (bias1, bias2, bias3, bias4)

    return a5, a, thetas, biases


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
