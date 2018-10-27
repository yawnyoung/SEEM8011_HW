import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.stats as st


def mul_ls_estimator(x, y):
    x_t = np.transpose(x)
    b = np.linalg.inv(x_t @ x) @ x_t @ y

    # print(b)

    return b


def calc_residuals(x, y, b):
    return y - x @ b


def calc_ssto(y):
    y_t = np.transpose(y)

    n = y.size

    ssto = y_t @ y - y_t @ np.ones((n, n)) @ y / n

    print('SSTO: ', ssto)

    return ssto


def calc_sse(x, y, b):
    e = calc_residuals(x, y, b)

    sse = np.transpose(e) @ e

    print('SSE: ', sse)

    return sse


def calc_mse(x, y, b):
    sse = calc_sse(x, y, b)

    p = x.shape[1]
    n = y.size

    mse = sse / (n - p)

    print('MSE: ', mse)

    return mse


def calc_estimated_bvar_mat(x, y, b):
    mse = calc_mse(x, y, b)

    estimated_var = mse * np.linalg.inv(np.transpose(x) @ x)

    print('Estimated variance-covariance matrix: \n', estimated_var)

    return estimated_var


def calc_estimated_yvar(x, y, b, xh):
    s2_b = calc_estimated_bvar_mat(x, y, b)
    s2_y = np.transpose(xh) @ s2_b @ xh

    print('estimated variance of y given {} is {}'.format(xh, s2_y))

    return s2_y


def calc_ssr(x, y, b):
    b_t = np.transpose(b)
    x_t = np.transpose(x)
    y_t = np.transpose(y)

    n = y.size

    ssr = b_t @ x_t @ y - y_t @ np.ones((n, n)) @ y / n

    print('SSR: ', ssr)

    return ssr


def calc_msr(x, y, b):
    ssr = calc_ssr(x, y, b)

    p = x.shape[1]

    msr = ssr / (p - 1)

    print('MSR: ', msr)

    return msr


def calc_R_sqr(x, y, b):
    ssto = calc_ssto(y)
    sse = calc_sse(x, y, b)

    r_sqr = 1 - sse / ssto

    print('the coefficient of determination R^2: ', r_sqr)

    return r_sqr


def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[x], data[y], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig


def pearson_correlation_coefficient(x, y):
    num = ((x - x.mean()) * (y - y.mean())).sum()
    den = np.sqrt(np.square(x - x.mean()).sum() * np.square(y - y.mean()).sum())
    r = num / den
    print('num: {}, den: {}, r: {}'.format(num, den, r))
    return r


def normal_probability_plot(x, y, b):

    n = y.size

    residuals = calc_residuals(x, y, b)

    order = residuals.argsort(axis=0)
    rank = order.argsort(axis=0) + 1

    ev_param = (rank - 0.375) / (n + 0.25)
    mse_sqrt = np.sqrt(calc_mse(x, y, b))
    ev = mse_sqrt * st.norm.ppf(ev_param)
    print(ev)

    plt.plot(ev, residuals, linestyle='none', marker='o', color='black', mfc='none')
    plt.xlabel('Expected')
    plt.ylabel('Residual')
    plt.show()


def breush_pagan_test(x, y, b):

    # regress the squared residuals
    e_sqr = np.square(calc_residuals(x, y, b))

    gamma = mul_ls_estimator(x, e_sqr)
    ssr_star = calc_ssr(x, e_sqr, gamma)
    print('ssr star: ', ssr_star)

    sse = calc_sse(x, y, b)

    n = y.size

    test_stat = (ssr_star / 2) / np.square(sse / n)

    print(st.chi2.ppf(0.99, 1))

    print(gamma)
    print('test stat: ', test_stat)


def calc_rap(x, y):
    b = mul_ls_estimator(x, y)

    n = y.size
    p = x.shape[1]

    sse = calc_sse(x, y, b)
    ssto = calc_ssto(y)

    rap = 1 - ((n - 1) / (n - p)) * (sse / ssto)

    return rap


def calc_cp(x, y, mse_full):
    b = mul_ls_estimator(x, y)

    n = y.size
    p = x.shape[1]

    sse = calc_sse(x, y, b)

    cp = sse / mse_full - (n - 2 * p)

    return cp


def calc_press(x, y):

    n = y.size

    press = 0

    for i in range(n):
        x_r = np.delete(x, i, 0)
        y_r = np.delete(y, i, 0)
        b_r = mul_ls_estimator(x_r, y_r)
        y_r_hat = x[i, :] @ b_r
        press += (y[i] - y_r_hat) * (y[i] - y_r_hat)

    return press


def calc_press_none(y):
    n = y.size

    press = 0

    for i in range(n):
        y_r = np.delete(y, i, 0)
        y_r_hat = y_r.mean()
        press += (y[i] - y_r_hat) * (y[i] - y_r_hat)

    return press


def plot_residuals(x, y):
    b = mul_ls_estimator(x, y)

    y_hat = x @ b

    e = calc_residuals(x, y, b)

    plt.plot(y_hat, e, linestyle='none', marker='o', color='black', mfc='none')
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.title('Residual plot')
    plt.show()
