import numpy as np
import math_utils
import matplotlib.pyplot as plt


def read_data(file_name = 'data_6_5.txt'):

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


def plot_residual_bp(residuals):
    fig, ax = plt.subplots()
    bp_dict = ax.boxplot(residuals)

    # print(bp_dict.keys())

    for line in bp_dict['medians']:
        x, y = line.get_xydata()[0]
        plt.text(x - 0.05, y, '%.1f' % y, verticalalignment='center', horizontalalignment='center')

    for line in bp_dict['boxes']:
        x, y = line.get_xydata()[0]
        plt.text(x - 0.05, y, '%.1f' % y, verticalalignment='center', horizontalalignment='center')
        x, y = line.get_xydata()[3]
        plt.text(x - 0.05, y, '%.1f' % y, verticalalignment='center', horizontalalignment='center')

    for line in bp_dict['caps']:
        x, y = line.get_xydata()[0]
        plt.text(x - 0.05, y, '%.1f' % y, verticalalignment='center', horizontalalignment='center')

    plt.ylabel('Residual')
    plt.show()


def plot_residuals(x, y, b):
    residuals = math_utils.calc_residuals(x, y, b)
    y_hat = x @ b

    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    axes[0, 0].plot(y_hat, residuals, linestyle='none', marker='o', color='black', mfc='none')
    axes[0, 0].set_xlabel('Fitted')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].set_title(r'Residual Plot against $\hat Y$')

    axes[0, 1].plot(x[:, 1], residuals, linestyle='none', marker='o', color='black', mfc='none')
    axes[0, 1].set_xlabel('Moisture Content')
    axes[0, 1].set_ylabel('Residual')
    axes[0, 1].set_title(r'Residual Plot against $X_1$')

    axes[1, 0].plot(x[:, 2], residuals, linestyle='none', marker='o', color='black', mfc='none')
    axes[1, 0].set_xlabel('Sweetness')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].set_title(r'Residual Plot against $X_2$')

    axes[1, 1].plot(x[:, 1] * x[:, 2], residuals, linestyle='none', marker='o', color='black', mfc='none')
    axes[1, 1].set_xlabel(r'$X_1X_2$')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title(r'Residual Plot against $X_1X_2$')

    plt.show()


def prob_6_6(x_arr, y_arr, b):
    mse = math_utils.calc_mse(x_arr, y_arr, b)
    msr = math_utils.calc_msr(x_arr, y_arr, b)

    f_star = msr / mse
    print('F*: ', f_star)

    estimated_var = math_utils.calc_estimated_bvar_mat(x_arr, y_arr, b)
    # print('s: \n', np.sqrt(estimated_var))


def prob_6_7(x_arr, y_arr, b):
    x1_arr = x_arr[:, :2]
    b_1 = math_utils.mul_ls_estimator(x1_arr, y_arr)
    print('b1: ', b_1)
    R_1 = math_utils.calc_R_sqr(x1_arr, y_arr, b_1)

    x2_arr = np.delete(x_arr, 1, 1)
    b_2 = math_utils.mul_ls_estimator(x2_arr, y_arr)
    print('b2: ', b_2)
    R_2 = math_utils.calc_R_sqr(x2_arr, y_arr, b_2)

    print(R_1 + R_2)


def prob_6_8(x_arr, y_arr, b):
    xh = np.array([1, 5, 4], np.newaxis)
    yh = xh @ b
    print('yh: ', yh)

    s2_y = math_utils.calc_estimated_yvar(x_arr, y_arr, b, xh)

    s_y = np.sqrt(s2_y)
    print('s_y: ', s_y)


def prob_6_5(x_arr, y_arr, b):

    n_data = y_arr.size
    n_var = 3

    data = np.ndarray((n_var, n_data), dtype=float)

    data[0] = y_arr.flatten()
    data[1] = x_arr[:, 1]
    data[2] = x_arr[:, 2]

    # fig = math_utils.scatterplot_matrix(data, ['brand liking', 'moisture content', 'sweetness'],
    #                                     linestyle='none', marker='o', color='black', mfc='none')
    #
    # fig.suptitle('Scatter plot matrix')
    # plt.show()

    # math_utils.pearson_correlation_coefficient(x_arr[:, 1], x_arr[:, 2])

    # plot_residuals(x_arr, y_arr, b)

    # math_utils.normal_probability_plot(x_arr, y_arr, b)

    math_utils.breush_pagan_test(x_arr, y_arr, b)


if __name__ == '__main__':
    x_arr, y_arr = read_data()
    b = math_utils.mul_ls_estimator(x_arr, y_arr)

    # prob_6_6(x_arr, y_arr, b)
    # prob_6_8(x_arr, y_arr, b)
    prob_6_5(x_arr, y_arr, b)