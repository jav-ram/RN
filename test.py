import numpy as np
import cv2
from PIL import Image
import PIL.ImageOps

import random

from feed_forward import feed_forward_two
from gradient_descent import gradient_descent
from cost_and_gradient import cost_and_gradient_two

# Load Variables set
theta = np.load('./dwv1.npy')
bias = np.load('./dbv1.npy')

# Load Test
k = np.asarray(cv2.imread('./out/House/1.jpg')).ravel()

if k.shape[0] != 784:
    k = cv2.resize(k, (int(28), int(28)))
b = (np.array(k).ravel() / 256).reshape(1, 784)

r = feed_forward_two(
    b,
    theta[0],
    theta[1],
    theta[2],
    bias[0],
    bias[1],
)


print(r[0])
