import numpy as np
import math_utils
import matplotlib.pyplot as plt


def read_data(file_name = 'data.txt'):

    with open(file_name) as f:
        y_list = []
        x_list = []
        for line in f:
            line = line.split()
            y_list.append(float(line[0]))
            x_list.append([float(line[1]), float(line[2])])

        y_arr = np.ndarray((len(y_list), 1))
        x_arr = np.ones((len(x_list), 3))

        for i in range(len(x_list)):
            x_arr[i][1] = x_list[i][0]
            x_arr[i][2] = x_list[i][1]
            y_arr[i] = y_list[i]

        return x_arr, y_arr


if __name__ == '__main__':
    x_arr, y_arr = read_data()
    b = math_utils.mul_ls_estimator(x_arr, y_arr)
    residuals = (math_utils.calc_residuals(x_arr, y_arr, b))
    fig, ax = plt.subplots()
    ax.boxplot(residuals)
    plt.show()
