import numpy as np
import os
import random
from PIL import Image
import PIL.ImageOps
import cv2

from sklearn.model_selection import train_test_split

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


def get_partitions(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    train = (x_train, y_train)
    test = (x_test, y_test)
    val = (x_val, y_val)

    return train, test, val


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

    circle_sample = name_to_array('./out/Circle/', circle_sample_names)
    egg_sample = name_to_array('./out/Egg/', egg_sample_names)
    house_sample = name_to_array('./out/House/', house_sample_names)
    mickey_sample = name_to_array('./out/MickeyMouse/', mickey_sample_names)
    question_sample = name_to_array('./out/QuestionMark/', question_sample_names)
    sad_sample = name_to_array('./out/SadFace/', sad_sample_names)
    smile_sample = name_to_array('./out/SmileyFace/', smile_sample_names)
    square_sample = name_to_array('./out/Square/', square_sample_names)
    tree_sample = name_to_array('./out/Tree/', tree_sample_names)
    triangle_sample = name_to_array('./out/Triangle/', triangle_sample_names)

    circle_partition = get_partitions(circle_sample, get_all_y_category(sample, CIRCLE, 10))
    egg_partition = get_partitions(egg_sample, get_all_y_category(sample, EGG, 10))
    house_partition = get_partitions(house_sample, get_all_y_category(sample, HOUSE, 10))
    mickey_partition = get_partitions(mickey_sample, get_all_y_category(sample, MICKEY, 10))
    question_partition = get_partitions(question_sample, get_all_y_category(sample, QUESTION, 10))
    sad_partition = get_partitions(sad_sample, get_all_y_category(sample, SAD, 10))
    smile_partition = get_partitions(smile_sample, get_all_y_category(sample, SMILEY, 10))
    square_partition = get_partitions(square_sample, get_all_y_category(sample, SQUARE, 10))
    tree_partition = get_partitions(tree_sample, get_all_y_category(sample, TREE, 10))
    triangle_partition = get_partitions(triangle_sample, get_all_y_category(sample, TRIANGLE, 10))

    train_x = np.vstack((
        circle_partition[0][0],
        egg_partition[0][0],
        house_partition[0][0],
        mickey_partition[0][0],
        question_partition[0][0],
        sad_partition[0][0],
        smile_partition[0][0],
        square_partition[0][0],
        tree_partition[0][0],
        triangle_partition[0][0],
    ))

    train_y = np.vstack((
        circle_partition[0][1],
        egg_partition[0][1],
        house_partition[0][1],
        mickey_partition[0][1],
        question_partition[0][1],
        sad_partition[0][1],
        smile_partition[0][1],
        square_partition[0][1],
        tree_partition[0][1],
        triangle_partition[0][1],
    ))

    test_x = np.vstack((
        circle_partition[1][0],
        egg_partition[1][0],
        house_partition[1][0],
        mickey_partition[1][0],
        question_partition[1][0],
        sad_partition[1][0],
        smile_partition[1][0],
        square_partition[1][0],
        tree_partition[1][0],
        triangle_partition[1][0],
    ))

    test_y = np.vstack((
        circle_partition[1][1],
        egg_partition[1][1],
        house_partition[1][1],
        mickey_partition[1][1],
        question_partition[1][1],
        sad_partition[1][1],
        smile_partition[1][1],
        square_partition[1][1],
        tree_partition[1][1],
        triangle_partition[1][1],
    ))

    val_x = np.vstack((
        circle_partition[2][0],
        egg_partition[2][0],
        house_partition[2][0],
        mickey_partition[2][0],
        question_partition[2][0],
        sad_partition[2][0],
        smile_partition[2][0],
        square_partition[2][0],
        tree_partition[2][0],
        triangle_partition[2][0],
    ))

    val_y = np.vstack((
        circle_partition[2][1],
        egg_partition[2][1],
        house_partition[2][1],
        mickey_partition[2][1],
        question_partition[2][1],
        sad_partition[2][1],
        smile_partition[1][1],
        square_partition[2][1],
        tree_partition[2][1],
        triangle_partition[1][1],
    ))

    return (train_x, train_y), (test_x, test_y), (val_x, val_y)


train, test, val = init_sample(784, 1000)

np.save('./train/train/x', train[0])
np.save('./train/train/y', train[1])

np.save('./train/test/x', test[0])
np.save('./train/test/y', test[1])

np.save('./train/validate/x', val[0])
np.save('./train/validate/y', val[1])
