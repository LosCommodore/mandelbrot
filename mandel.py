# https://de.wikipedia.org/wiki/Mandelbrot-Menge
import numpy as np
import matplotlib.pyplot as plt


def mandel(c, limit=1000, max_iter=255):
    value = 0 + 0j
    for i in range(max_iter):
        value = value ** 2 + c
        if abs(value) > limit:
            return i

    return max_iter


if __name__ == "__main__":
    # width, height = 1024, 768
    width, height = 3840, 2160

    pic = np.zeros((height, width), dtype="int")
    zoom_x, zoom_y = width / 4, height / 4
    shift_x, shift_y = - width / 2, -height / 2

    for y, x in np.ndindex(pic.shape):
        c = (x + shift_x) / zoom_x + (y + shift_y) * 1j / zoom_y
        pic[y, x] = mandel(c)

    cm = 1 / 2.54  # centimeters in inches
    fig = plt.figure(figsize=(20 * cm, 15 * cm))
    plt.imshow(pic, vmin=0, vmax=10, )
    plt.axis('off')
    plt.show()
