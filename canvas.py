from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import cv2


from feed_forward import feed_forward_two

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

types = ['circulo', 'huevo', 'casa', 'mickey', 'signo', 'triste', 'feliz', 'cuadrado', 'arbol', 'triangulo']

def save():
    global image_number
    filename = 'image.bmp'   # image_number increments by 1 at every save
    image1.save(filename)

    basewidth = 28
    img = Image.open(filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(filename)

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

    for i in range(10):
        print(types[i] + '\t', r[0][i].ravel())


def eraser():
    cv.delete("all")
    draw.rectangle((0, 0, 560, 560), "white")

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(event):
    x1, y1 = (event.x), (event.y)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval((x1, y1, x2, y2), fill='black', width=10)
    #  --- PIL
    draw.line((x1, y1, x2, y2), fill='black', width=45)


root = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=500, height=500, bg='white')
# --- PIL
image1 = PIL.Image.new('RGB', (500, 500), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()

btn_restart = Button(text="restart", command=eraser)
btn_restart.pack()

root.mainloop()