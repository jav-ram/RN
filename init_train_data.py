import numpy as np
import os
import random
from PIL import Image
import PIL.ImageOps
import cv2

out_dir = './out/'

CIRCLE = 0
EGG = 1
HOUSE = 2
MICKEY = 3
QUESTION = 4
SAD = 5
SMILEY = 6
SQUARE = 7
TREE = 8
TRIANGLE = 9


def alpha(x):
    if x / 255 > 0.5:
        return 1
    else:
        return 0


valpha = np.vectorize(alpha)


def name_to_array(directory, names):
    a = []
    for n in names:
        k = np.asarray(cv2.imread(directory + n, cv2.IMREAD_GRAYSCALE)).ravel()

        if k.shape[0] != 784:
            k = cv2.resize(k, (int(28), int(28)))
        b = (255 - np.array(k).ravel()) / 255
        a.append(b)

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
    egg_sample_names = random.choices(os.listdir('./out/Egg/'), k=sample)
    house_sample_names = random.choices(os.listdir('./out/House/'), k=sample)
    mickey_sample_names = random.choices(os.listdir('./out/MickeyMouse/'), k=sample)
    question_sample_names = random.choices(os.listdir('./out/QuestionMark/'), k=sample)
    sad_sample_names = random.choices(os.listdir('./out/SadFace/'), k=sample)
    smile_sample_names = random.choices(os.listdir('./out/SmileyFace/'), k=sample)
    square_sample_names = random.choices(os.listdir('./out/Square/'), k=sample)
    tree_sample_names = random.choices(os.listdir('./out/Tree/'), k=sample)
    triangle_sample_names = random.choices(os.listdir('./out/Triangle/'), k=sample)

    X = np.vstack((
        name_to_array('./out/Circle/', circle_sample_names),
        name_to_array('./out/Egg/', egg_sample_names),
        name_to_array('./out/House/', house_sample_names),
        name_to_array('./out/MickeyMouse/', mickey_sample_names),
        name_to_array('./out/QuestionMark/', question_sample_names),
        name_to_array('./out/SadFace/', sad_sample_names),
        name_to_array('./out/SmileyFace/', smile_sample_names),
        name_to_array('./out/Square/', square_sample_names),
        name_to_array('./out/Tree/', tree_sample_names),
        name_to_array('./out/Triangle/', triangle_sample_names)
    ))

    print(X.shape)

    Y = np.vstack((
        get_all_y_category(sample, CIRCLE, 10),
        get_all_y_category(sample, EGG, 10),
        get_all_y_category(sample, HOUSE, 10),
        get_all_y_category(sample, MICKEY, 10),
        get_all_y_category(sample, QUESTION, 10),
        get_all_y_category(sample, SAD, 10),
        get_all_y_category(sample, SMILEY, 10),
        get_all_y_category(sample, SQUARE, 10),
        get_all_y_category(sample, TREE, 10),
        get_all_y_category(sample, TRIANGLE, 10)
    ))

    return X, Y


x, y = init_sample(784, 1000)

np.save('./train/Xv1', x)
np.save('./train/Yv1', y)
