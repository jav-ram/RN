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


# load thetas and bias
theta = np.load('./theta.npy')
bias = np.load('./bias.npy')

# TRAIN
# load data
x_train = np.load('./train/train/x.npy')
y_train = np.load('./train/train/y.npy')
# feed forward
r = feed_forward_two(
    x_train,
    theta[0],
    theta[1],
    theta[2],
    theta[3],
    bias[0],
    bias[1],
    bias[2],
    bias[3],
)
# accuracy
h_train_answer = get_answer(r[0].T)
y_train_answer = get_answer(y_train)
print(get_accuracy(h_train_answer, y_train_answer))

# TEST
# load data
x_test = np.load('./train/test/x.npy')
y_test = np.load('./train/test/y.npy')
# feed forward
r = feed_forward_two(
    x_test,
    theta[0],
    theta[1],
    theta[2],
    theta[3],
    bias[0],
    bias[1],
    bias[2],
    bias[3],
)
# accuracy
h_test_answer = get_answer(r[0].T)
y_test_answer = get_answer(y_test)
print(get_accuracy(h_test_answer, y_test_answer))

# VALIDATE
# load data
x_val = np.load('./train/validate/x.npy')
y_val = np.load('./train/validate/y.npy')
# feed forward
r = feed_forward_two(
    x_val,
    theta[0],
    theta[1],
    theta[2],
    theta[3],
    bias[0],
    bias[1],
    bias[2],
    bias[3],
)
# accuracy
h_val_answer = get_answer(r[0].T)
y_val_answer = get_answer(y_val)
print(get_accuracy(h_val_answer, y_val_answer))
