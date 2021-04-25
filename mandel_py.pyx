import numpy as np

cdef int mandel(double complex c, double limit, int max_iter):
    cdef double complex value = 0 + 0j
    cdef int i

    for i in range(max_iter, 1, -1):
        value = value ** 2 + c
        if abs(value) > limit:
            return i

    return 0


def generate_pic_old(width, height,z_x ,z_y,t_x,t_y,int max_val, int max_iter):
    pic = np.zeros((height, width), dtype="int")
    zoom_x, zoom_y = width * z_x, height * z_y
    shift_x, shift_y = - width * t_x, -height * t_y

    for y, x in np.ndindex(pic.shape):
        c = (x + shift_x) / zoom_x + (y + shift_y) * 1j / zoom_y
        pic[y, x] = mandel(c, max_val, max_iter)

    return pic


def generate_pic(width, height, z_x, z_y, t_x, t_y, max_val, max_iter):
    pic = np.zeros((height, width), dtype="int")
    zoom_x, zoom_y = width * z_x, height * z_y
    shift_x, shift_y = - width * t_x, -height * t_y

    X = np.linspace(shift_x / zoom_x, (shift_x + width)  / zoom_x, width,  dtype=complex)
    Y = np.linspace(shift_y / zoom_y, (shift_y + height) / zoom_y, height, dtype=complex) * 1j

    C = X[:, np.newaxis] + Y[np.newaxis, :]

    for (x, y), c in np.ndenumerate(C):
        pic[y, x] = mandel(c, max_val, max_iter)

    return pic