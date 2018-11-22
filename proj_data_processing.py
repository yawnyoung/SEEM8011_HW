import numpy as np
from numpy import genfromtxt

if __name__ == '__main__':

    proj_data = genfromtxt('modified_pure_proj_data.csv', delimiter=',')

    num_days = proj_data.shape[0]
    num_vars = proj_data.shape[1] - 1
    period = 3

    new_data = np.ndarray(shape=(num_days - period, period * num_vars + 1), dtype=float)

    new_ridx = 0

    for i in range(period, num_days):

        new_data[new_ridx][0] = proj_data[i][0]

        var_list = []
        for j in range(i - 3, i):
            for value in proj_data[j][1:]:
                var_list.append(value)

        for k in range(period * num_vars):
            new_data[new_ridx][k+1] = var_list[k]

        new_ridx += 1

    print(new_data)

    np.savetxt('data.csv', new_data, delimiter=',')
