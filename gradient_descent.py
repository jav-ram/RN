import numpy as np
from PIL import Image
import PIL.ImageOps


from cost_and_gradient import cost_and_gradient_two
from feed_forward import feed_forward_two

norm = lambda v: ((v ** 2.0).sum()) ** 0.5


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


v = np.array(PIL.Image.open('./out/Circle/1.jpg').convert("L")).ravel().reshape((1, 784)) / 255
v1 = np.zeros((1, 784))

def gradient_descent(
        X,
        y,
        theta_0,
        bias_0,
        cost_and_gradient,
        alpha=0.0000000000001,
        beta=0.0000000001,
        threshold=1000000,
        max_iter=1000000):
    theta, bias, last_cost, i = theta_0, bias_0, 999999999999, 0
    Dw = [np.zeros(theta1Size), np.zeros(theta2Size), np.zeros(theta3Size)]
    Db = [0.0] * 38
    # alpha = np.random.uniform(low=0, high=alpha, size=thetasLen)
    # beta = np.random.uniform(low=0, high=beta, size=biasLen)
    while i < max_iter and abs(cost_and_gradient(X, y, theta, bias, Dw)[0]) > threshold:
        cost, gradient_w, gradient_b, gradient = cost_and_gradient(X, y, theta, bias, Dw)

        theta -= alpha * gradient_w
        bias -= beta * gradient_b
        # np.abs(cost) == np.inf or print(abs(cost), norm(cost_and_gradient(X, y, theta, bias, Dw)[-1]))

        theta_array = np.split(theta, [theta1Len, theta1Len + theta2Len, theta1Len + theta2Len + theta3Len])
        t1 = theta_array[0].reshape(theta1Size)
        t2 = theta_array[1].reshape(theta2Size)
        t3 = theta_array[2].reshape(theta3Size)

        bias_array = np.split(bias, [bias1Len, bias1Len + bias2Len, bias1Len + bias2Len + bias3Len])
        b1 = bias_array[0].reshape(bias1Size)
        b2 = bias_array[1].reshape(bias2Size)
        b3 = bias_array[2].reshape(bias3Size)
        r1 = feed_forward_two(
            X,
            t1,
            t2,
            t3,
            b1,
            b2,
            b3,
        )[0].T[1000]
        print(r1)

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

t1 = np.random.rand(16, 784)
t2 = np.random.rand(16, 16)
t3 = np.random.rand(6, 16)

b1 = np.random.rand(16, 1)
b2 = np.random.rand(16, 1)
b3 = np.random.rand(6, 1)

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
