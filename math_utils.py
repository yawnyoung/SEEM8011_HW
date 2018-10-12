import numpy as np


def mul_ls_estimator(x, y):
    x_t = np.transpose(x)
    x_t_times_x = np.matmul(x_t, x)
    b = np.linalg.inv(x_t_times_x)
    b = np.matmul(b, x_t)
    b = np.matmul(b, y)

    # print(b)

    return b


def calc_residuals(x, y, b):
    return y - np.matmul(x, b)