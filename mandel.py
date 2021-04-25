# https://de.wikipedia.org/wiki/Mandelbrot-Menge
import numpy as np
import matplotlib.pyplot as plt
import mandel_py
import numba


@numba.njit()
def mandel(c, limit=2, max_iter=255):
    value = 0 + 0j
    for i in range(max_iter, 1, -1):
        value = value ** 2 + c
        if abs(value) > limit:
            return i

    return 0


@numba.njit()
def generate_pic(width, height, z_x, z_y, t_x, t_y, max_val, max_iter):
    pic = np.zeros((height, width), dtype="int")
    zoom_x, zoom_y = width * z_x, height * z_y
    shift_x, shift_y = - width * t_x, -height * t_y

    X = np.linspace(shift_x / zoom_x, (shift_x + width) / zoom_x, width)
    X = np.reshape(X, (-1, 1))
    Y = np.linspace(shift_y / zoom_y, (shift_y + height) / zoom_y, height) * 1j
    Y = np.reshape(Y, (1, -1))
    C = X + Y

    for (x, y), C in np.ndenumerate(C):
        pic[y, x] = mandel(C, max_val, max_iter)

    return pic


if __name__ == "__main__":
    # width, height = 1024, 768
    # width, height = 3840, 2160
    width, height = 3840, 2160

    max_iter = 255
    pic = generate_pic(width, height, z_x=2 / 5, z_y=2 / 5, t_x=4 / 5, t_y=1 / 2, max_val=2, max_iter=max_iter)

    # noinspection PyUnresolvedReferences
    if only_save := True:
        plt.imsave('mandel.png', pic, vmin=0, vmax=255, cmap=plt.cm.jet)
    else:
        cm = 1 / 2.54  # centimeters in inches
        fig = plt.figure(figsize=(20 * cm, 15 * cm), dpi=100)
        plt.imshow(pic, vmin=0, vmax=max_iter, cmap=plt.get_cmap("rainbow"))
        plt.axis('off')
        plt.savefig('mandel.png', dpi=100)
        plt.show()
