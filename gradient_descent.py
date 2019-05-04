import numpy as np
from PIL import Image
import PIL.ImageOps


from cost_and_gradient import cost_and_gradient_two
from feed_forward import feed_forward_two

norm = lambda v: ((v ** 2.0).sum()) ** 0.5


theta1Size = (24, 784)
theta1Len = theta1Size[0] * theta1Size[1]

theta2Size = (20, 24)
theta2Len = theta2Size[0] * theta2Size[1]

theta3Size = (16, 20)
theta3Len = theta3Size[0] * theta3Size[1]

theta4Size = (10, 16)
theta4Len = theta4Size[0] * theta4Size[1]

thetasLen = theta1Len + theta2Len + theta3Len + theta4Len

v = np.array(PIL.Image.open('./out/Circle/1.jpg').convert("L")).ravel().reshape((1, 784)) / 255
v1 = np.zeros((1, 784))

def gradient_descent(
        X,
        y,
        theta_0,
        bias_0,
        cost_and_gradient,
        alpha=0.0000000000001,
        threshold=1000000,
        max_iter=1000000):
    theta, bias, last_cost, i = theta_0, bias_0, 999999999999, 0
    Dw = [np.zeros(theta1Size), np.zeros(theta2Size), np.zeros(theta3Size), np.zeros(theta4Size)]
    # alpha = np.random.uniform(low=0, high=alpha, size=thetasLen)
    # beta = np.random.uniform(low=0, high=beta, size=biasLen)
    while i < max_iter and abs(cost_and_gradient(X, y, theta, bias, Dw)[0]) > threshold:
        cost, gradient_w, gradient_b, gradient = cost_and_gradient(X, y, theta, bias, Dw)

        theta = (
            theta[0] + alpha * gradient_w[0],
            theta[1] + alpha * gradient_w[1],
            theta[2] + alpha * gradient_w[2],
            theta[3] + alpha * gradient_w[3],
        )

        bias = (
            bias[0] + alpha * gradient_b[0],
            bias[1] + alpha * gradient_b[1],
            bias[2] + alpha * gradient_b[2],
            bias[3] + alpha * gradient_b[3],
        )

        r1 = feed_forward_two(
            np.vstack((
                X[0],
                X[1001],
                X[2002],
                X[3003],
                X[4003],
            )),
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            bias[0],
            bias[1],
            bias[2],
            bias[3],
        )
        # print(r1[0].T[0])
        print()
        print((r1[0].T[0].T))
        print((r1[0].T[1].T))
        print((r1[0].T[2].T))
        print((r1[0].T[3].T))

        np.abs(cost) == np.inf or print(abs(cost), norm(cost_and_gradient(X, y, theta, bias, Dw)[-1]))

        Dw = gradient_w
        i += 1
    print(abs(cost_and_gradient(X, y, theta, bias, Dw)[0]))
    return theta, bias


x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

# t1 = np.random.rand(16, 784)
# t2 = np.random.rand(16, 16)
# t3 = np.random.rand(6, 16)
#
# b1 = np.random.rand(16, 1)
# b2 = np.random.rand(16, 1)
# b3 = np.random.rand(6, 1)

# t1 = np.array([
#     [-0.08152351,  0.59674655],
#     [ -0.27872879,  0.28340382],
#     [ -0.33741438, -0.20879898],
# ])
# t2 = np.array([
#     [ 0.26758386,  0.35642368, -0.12886719]
# ])
#
# b1 = np.array([
#     [0.07838552],
#     [0.11359833],
#     [-0.12019645]
# ])
# b2 = np.array([
#     [-0.89393996]
# ])

y = np.array([
    [1],
    [0],
    [1],
    [0]
])

# x = np.array([[1.0], [2.0], [3.0]])
# t1 = np.array([[1.0], [0.0]])
# t2 = np.array([[0.0, 2.0]])
# y = np.array([[0.0], [0.0], [0.0]])

# r, t, g, b = feed_forward_two(x, t1, t2, b1, b2)
#
# print(np.nan_to_num(r))

# print(np.hstack((b1.ravel(), b2.ravel(), b3.ravel())).shape)
#
# print(gradient_descent(
#         x,
#         y,
#         np.hstack((t1.ravel(), t2.ravel(), t3.ravel())),
#         np.hstack((b1.ravel(), b2.ravel(), b3.ravel())),
#         cost_and_gradient_two,
# ))
