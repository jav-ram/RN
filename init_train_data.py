import numpy as np
import os
import random
from PIL import Image
import PIL.ImageOps

out_dir = './out/'

CIRCLE = 0
HOUSE = 1
SMILE = 2
SQUARE = 3
TREE = 4
TRIANGLE = 5
SAD = 6
EGG = 7
MOUSE = 8
QUESTION = 9


def alpha(x):
    if x / 255 > 0.5:
        return 1
    else:
        return 0


valpha = np.vectorize(alpha)


def name_to_array(directory, names):
    a = []
    for n in names:
        b = np.array(PIL.Image.open(directory + n).convert("L")).ravel()
        a.append(valpha(b))

    return np.asarray(a)#.astype(np.float128)


def get_y(category, classes=10):
    y = np.arange(0, classes).reshape(1, classes)

    y = np.ma.masked_where(y == category, y).mask.astype(int)
    return y


def get_all_y_category(sample, category, classes=10):
    Y = []
    for i in range(0, sample):
        y = get_y(category, classes)
        Y.append(y)

    Y = np.asarray(Y).reshape((sample, classes))
    return Y


def init_sample(features, sample):
    circle_sample_names = random.choices(os.listdir('./out/Circle/'), k=sample)
    # house_sample_names = random.choices(os.listdir('./out/House/'), k=sample)
    # smile_sample_names = random.choices(os.listdir('./out/Smiley Face/'), k=sample)
    # square_sample_names = random.choices(os.listdir('./out/Square/'), k=sample)
    # tree_sample_names = random.choices(os.listdir('./out/Tree/'), k=sample)
    # triangle_sample_names = random.choices(os.listdir('./out/Triangle/'), k=sample)

    X = np.vstack((
        name_to_array('./out/Circle/', circle_sample_names),
        # name_to_array('./out/House/', house_sample_names),
        # name_to_array('./out/Smiley Face/', smile_sample_names),
        # name_to_array('./out/Square/', square_sample_names),
        # name_to_array('./out/Tree/', tree_sample_names),
        # name_to_array('./out/Triangle/', triangle_sample_names)
    ))

    print(X.shape)

    Y = np.vstack((
        get_all_y_category(sample, CIRCLE, 6),
        # get_all_y_category(sample, HOUSE, 6),
        # get_all_y_category(sample, SMILE, 6),
        # get_all_y_category(sample, SQUARE, 6),
        # get_all_y_category(sample, TREE, 6),
        # get_all_y_category(sample, TRIANGLE, 6)
    ))

    return X, Y


x, y = init_sample(784, 1000)

np.save('./train/X', x)
np.save('./train/Y', y)
