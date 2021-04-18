# https://de.wikipedia.org/wiki/Mandelbrot-Menge
import numpy as np
import matplotlib.pyplot as plt
import mandel_py


def mandel(c, limit=1000, max_iter=255):
    value = 0 + 0j
    for i in range(max_iter, 1, -1):
        value = value ** 2 + c
        if abs(value) > limit:
            return i

    return 0


def generate_pic(width, height):
    pic = np.zeros((height, width), dtype="int")
    zoom_x, zoom_y = width / 4, height / 4
    shift_x, shift_y = - width * 2 / 3, -height * 1 / 2
    for y, x in np.ndindex(pic.shape):
        c = (x + shift_x) / zoom_x + (y + shift_y) * 1j / zoom_y
        pic[y, x] = mandel_py.mandel(c)

    return pic


if __name__ == "__main__":
    #width, height = 1024, 768
    #width, height = 8*3840, 8*2160
    width, height = 16 * 3840, 16 * 2160

    pic = mandel_py.generate_pic(width, height)

    # noinspection PyUnresolvedReferences
    if only_save := True:
        plt.imsave('mandel.png', pic, vmin=0,vmax=255, cmap=plt.cm.jet)
    else:
        cm = 1 / 2.54  # centimeters in inches
        fig = plt.figure(figsize=(20 * cm, 15 * cm), dpi=100)
        plt.imshow(pic, vmin=0, vmax=255, cmap=plt.get_cmap("rainbow"))
        plt.axis('off')
        plt.savefig('mandel.png', dpi=100)
        plt.show()

