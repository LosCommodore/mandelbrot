import numpy as np

cdef int mandel(double complex c, double limit=1000, int max_iter=255):
    cdef double complex value = 0 + 0j
    cdef int i

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
        pic[y, x] = mandel(c)

    return pic