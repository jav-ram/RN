from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import cv2


from feed_forward import feed_forward_two


def save():
    global image_number
    filename = 'image.bmp'   # image_number increments by 1 at every save
    image1.save(filename)

    k = np.asarray(cv2.imread(filename, cv2.IMREAD_GRAYSCALE)).ravel()

    if k.shape[0] != 784:
        k = cv2.resize(k, (int(28), int(28)))
    b = ((255 - np.array(k).ravel()) / 255).reshape(1, 784)
    print(b.shape)

    # load thetas and bias
    theta = np.load('./theta.npy')
    bias = np.load('./bias.npy')

    r = feed_forward_two(
        b,
        theta[0],
        theta[1],
        theta[2],
        bias[0],
        bias[1],
        bias[2],
    )

    print(r[0])




def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(event):
    x1, y1 = (event.x), (event.y)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval((x1, y1, x2, y2), fill='black', width=5)
    #  --- PIL
    draw.line((x1, y1, x2, y2), fill='black', width=5)


root = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=600, height=600, bg='white')
# --- PIL
image1 = PIL.Image.new('RGB', (600, 600), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()

root.mainloop()