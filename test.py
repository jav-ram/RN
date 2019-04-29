import numpy as np
from PIL import Image
import PIL.ImageOps

import random

from feed_forward import feed_forward_two
from gradient_descent import gradient_descent
from cost_and_gradient import cost_and_gradient_two

# Load Variables set
theta = np.load('./dw.npy')
bias = np.load('./db.npy')

# Load Test
img = np.array(PIL.Image.open('./out/Triangle/12.jpg').convert("L")).ravel().reshape((1, 784)).astype(np.float) / 255
# v1 = np.array(PIL.Image.open('./out/Circle/2.jpg').convert("L")).ravel().reshape((1, 784)).astype(np.float) / 255

r = feed_forward_two(
    img,
    theta[0],
    theta[1],
    theta[2],
    bias[0],
    bias[1],
)


print(r[0])
