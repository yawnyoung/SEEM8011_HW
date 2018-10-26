import numpy as np
import math_utils
import matplotlib.pyplot as plt


def read_data6_5():
    file_name = 'data_6_5.txt'
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


def prob_7_3():
    # read data
    x_arr, y_arr = read_data6_5()

    # regression with x1 and x2
    b_12 = math_utils.mul_ls_estimator(x_arr, y_arr)
    ssr_12 = math_utils.calc_ssr(x_arr, y_arr, b_12)
    print('ssr_12: ', ssr_12)
    sse_12 = math_utils.calc_sse(x_arr, y_arr, b_12)
    print('sse_12: ', sse_12)
    ssto_12 = math_utils.calc_ssto(y_arr)
    mse_12 = math_utils.calc_mse(x_arr, y_arr, b_12)
    print('mse_12: ', mse_12)

    # regression with x1
    b_1 = math_utils.mul_ls_estimator(x_arr[:, :2], y_arr)
    print('b1: ', b_1)
    ssr_1 = math_utils.calc_ssr(x_arr[:, :2], y_arr, b_1)
    print('ssr_1: ', ssr_1)
    sse_1 = math_utils.calc_sse(x_arr[:, :2], y_arr, b_1)
    print('sse_1: ', sse_1)

    # regression with x2
    x2_arr = np.delete(x_arr, 1, 1)
    b_2 = math_utils.mul_ls_estimator(x2_arr, y_arr)
    ssr_2 = math_utils.calc_ssr(x2_arr, y_arr, b_2)
    print('ssr 1 after 2: ', ssr_12 - ssr_2)
    sse_2 = math_utils.calc_sse(x2_arr, y_arr, b_2)
    print('sse_2: ', sse_2)

    print(ssr_12 - ssr_1)

    print(y_arr.size)





if __name__ == '__main__':
    prob_7_3()