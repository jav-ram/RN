import numpy as np
from feed_forward import feed_forward_two

def get_answer(h):
    m, n = h.shape
    r = []
    for i in range(m):
        r.append(np.argmax(h[i]))

    return np.asarray(r)


def get_accuracy(h, y):
    m, = h.shape
    r = []
    for i in range(m):
        if h[i] == y[i]:
            r.append(1)
        else:
            r.append(0)
    r = np.asarray(r)

    return r.sum() / len(r)

def print_accuracy(type, theta, bias):
    # load data
    x = np.load('./train/' + type + '/x.npy')
    y = np.load('./train/' + type + '/y.npy')
    # feed forward
    r = feed_forward_two(
        x,
        theta[0],
        theta[1],
        theta[2],
        bias[0],
        bias[1],
        bias[2],
    )
    # accuracy
    h = get_answer(r[0].T)
    y_answer = get_answer(y)
    print(type + '\t', get_accuracy(h, y_answer))

# load thetas and bias
theta = np.load('./theta.npy')
bias = np.load('./bias.npy')

# TRAIN
print_accuracy('train', theta, bias)

# TEST
print_accuracy('test', theta, bias)

# VALIDATE
print_accuracy('val', theta, bias)
