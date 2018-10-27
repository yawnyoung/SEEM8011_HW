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


def read_data_6_15():
    file_name = 'data_6_15.txt'
    with open(file_name) as f:
        y_list = []
        x_list = []
        for line in f:
            line = line.split()
            y_list.append(float(line[0]))
            x_list.append([float(line[1]), float(line[2]), float(line[3])])

        y_arr = np.ndarray((len(y_list), 1))
        x_arr = np.ones((len(x_list), 4))

        for i in range(len(x_list)):
            x_arr[i][1] = x_list[i][0]
            x_arr[i][2] = x_list[i][1]
            x_arr[i][3] = x_list[i][2]
            y_arr[i] = y_list[i]

        # print('y: ', y_arr)
        # print('x: ', x_arr)

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


def prob_9_9():
    # read data
    x_arr, y_arr = read_data_6_15()

    """
    Models
    """
    n = y_arr.size
    # print('n: ', n)

    # full model
    b_123 = math_utils.mul_ls_estimator(x_arr, y_arr)
    # print('b_123: ', b_123)
    # sse_123 = math_utils.calc_sse(x_arr, y_arr, b_123)
    # print('sse_123: ', sse_123)
    mse_123 = math_utils.calc_mse(x_arr, y_arr, b_123)
    # print('mse_123: ', mse_123)
    # rap_123 = math_utils.calc_rap(x_arr, y_arr)
    # print('rap_123: ', rap_123)
    # cp_123 = math_utils.calc_cp(x_arr, y_arr, mse_123)
    # print('cp_123: ', cp_123)
    # press_123 = math_utils.calc_press(x_arr, y_arr)
    # print('press_123: ', press_123)

    # model none sse = ssto
    # ssto = math_utils.calc_ssto(y_arr)
    # print('y mean: ', y_arr.mean())
    # cp_none = ssto / mse_123 - (n - 2 * 1)
    # print('cp_none: ', cp_none)
    # press_none = math_utils.calc_press_none(y_arr)
    # print('press_none: ', press_none)

    # model x1
    # x_arr_r = x_arr[:, :2]

    # model x2
    # x_arr_r = np.delete(x_arr, 1, 1)
    # x_arr_r = x_arr_r[:, :2]

    # model x3
    # x_arr_r = np.delete(x_arr, 1, 1)
    # x_arr_r = np.delete(x_arr_r, 1, 1)

    # model x12
    # x_arr_r = x_arr[:, :3]

    # model x13
    # x_arr_r = np.delete(x_arr, 2, 1)

    # model x23
    # x_arr_r = np.delete(x_arr, 1, 1)
    # print(x_arr_r)

    # model x123
    x_arr_r = x_arr

    # b_r = math_utils.mul_ls_estimator(x_arr_r, y_arr)
    # print('b_r: ', b_r)

    # sse_r = math_utils.calc_sse(x_arr_r, y_arr, b_r)
    # rap_r = math_utils.calc_rap(x_arr_r, y_arr)
    # print('rap_r: ', rap_r)
    # cp_r = math_utils.calc_cp(x_arr_r, y_arr, mse_123)
    # print('cp_r: ', cp_r)
    # press_r = math_utils.calc_press(x_arr_r, y_arr)
    # print('press_r: ', press_r)

    # math_utils.plot_residuals(x_arr_r, y_arr)

    n_var = 4
    data = np.ndarray((n_var, n), dtype=float)

    data[0] = y_arr.flatten()
    data[1] = x_arr[:, 1]
    data[2] = x_arr[:, 2]
    data[3] = x_arr[:, 3]

    fig = math_utils.scatterplot_matrix(data, ['Y', '$X_1$', '$X_2$', '$X_3$'],
                                        linestyle='none', marker='o', color='black', mfc='none')

    fig.suptitle('Scatter plot matrix')
    plt.show()


if __name__ == '__main__':
    # prob_7_3()
    prob_9_9()