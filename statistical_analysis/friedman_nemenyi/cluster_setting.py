# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
np.set_printoptions(suppress=True)
import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import chi2
pd.set_option('display.max_columns', None)

def ranks(data: np.array, descending=True):
    """ Computes the rank of the elements in data.
    :param data: 2-D matrix
    :param descending: boolean (default False). If true, rank is sorted in descending order.
    :return: ranks, where ranks[i][j] == rank of the i-th row w.r.t the j-th column.
    """
    s = 0 if (descending is False) else 1

    # Compute ranks. (ranks[i][j] == rank of the i-th treatment on the j-th sample.)
    if data.ndim == 2:
        ranks = np.ones(data.shape)
        for i in range(data.shape[0]):
            values, indices, rep = np.unique(
                (-1) ** s * np.sort((-1) ** s * data[i, :]), return_index=True, return_counts=True, )
            for j in range(data.shape[1]):
                ranks[i, j] += indices[values == data[i, j]] + \
                               0.5 * (rep[values == data[i, j]] - 1)
        #print(np.mean(ranks,axis=0))
        return ranks
    elif data.ndim == 1:
        ranks = np.ones((data.size,))
        values, indices, rep = np.unique(
            (-1) ** s * np.sort((-1) ** s * data), return_index=True, return_counts=True, )
        for i in range(data.size):
            ranks[i] += indices[values == data[i]] + \
                        0.5 * (rep[values == data[i]] - 1)
        return ranks


def sign_test(data):
    """ Given the results drawn from two algorithms/methods X and Y, the sign test analyses if
    there is a difference between X and Y.
    .. note:: Null Hypothesis: Pr(X<Y)= 0.5
    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value from the binomial distribution.
    :return bstat: Number of successes.
    """

    if type(data) == pd.DataFrame:
        data = data.values

    if data.shape[1] == 2:
        X, Y = data[:, 0], data[:, 1]
        n_perf = data.shape[0]
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of dimensions for axis 1')

    # Compute the differences
    Z = X - Y
    # Compute the number of pairs Z<0
    Wminus = sum(Z < 0)
    # If H_0 is true ---> W follows Binomial(n,0.5)
    p_value_minus = 1 - binom.cdf(k=Wminus, p=0.5, n=n_perf)

    # Compute the number of pairs Z>0
    Wplus = sum(Z > 0)
    # If H_0 is true ---> W follows Binomial(n,0.5)
    p_value_plus = 1 - binom.cdf(k=Wplus, p=0.5, n=n_perf)

    p_value = 2 * min([p_value_minus, p_value_plus])

    return pd.DataFrame(data=np.array([Wminus, Wplus, p_value]), index=['Num X<Y', 'Num X>Y', 'p-value'],
                        columns=['Results'])

def friedman_test(data):
    """ Friedman ranking test.
    ..note:: Null Hypothesis: In a set of k (>=2) treaments (or tested algorithms), all the treatments are equivalent, so their average ranks should be equal.
    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value.
    :return friedman_stat: Friedman's chi-square.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        data = data.values

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    # Compute ranks.
    datarank = ranks(data)
    #print(datarank)

    # Compute for each algorithm the ranking average.
    avranks = np.mean(datarank, axis=0)
    print(avranks)

    # Get Friedman statistics
    friedman_stat = (12.0 * n_samples) / (k * (k + 1.0)) * \
                    (np.sum(avranks ** 2) - (k * (k + 1) ** 2) / 4.0)

    # Compute p-value
    p_value = (1.0 - chi2.cdf(friedman_stat, df=(k - 1)))

    return pd.DataFrame(data=np.array([friedman_stat, p_value]), index=['Friedman-statistic', 'p-value'],
                        columns=['Results'])
timeframe = 't-3'
classifier = 'mlp'
kernel = 'rbf'
C = 0.1
cluster = [2,4,6,8,10,12,14,16]

metrics = ['F2','Auc','G-mean','J']
filepath = '../../parameter/setting_result/' + classifier + '/' + timeframe + '/' + kernel + '/' + str(C) + '_'
root_container = [filepath + str(i) + '.xlsx' for i in cluster]
data_container = [pd.read_excel(path).set_index('Unnamed: 0') for path in root_container]

f2_2, auc_2, g_2, j_2  = data_container[0].loc['F2',:], data_container[0].loc['Auc',:], data_container[0].loc['G-mean',:], data_container[0].loc['J',:]
f2_4, auc_4, g_4, j_4  = data_container[1].loc['F2',:], data_container[1].loc['Auc',:], data_container[1].loc['G-mean',:], data_container[1].loc['J',:]
f2_6, auc_6, g_6, j_6  = data_container[2].loc['F2',:], data_container[2].loc['Auc',:], data_container[2].loc['G-mean',:], data_container[2].loc['J',:]
f2_8, auc_8, g_8, j_8  = data_container[3].loc['F2',:], data_container[3].loc['Auc',:], data_container[3].loc['G-mean',:], data_container[3].loc['J',:]
f2_10, auc_10, g_10, j_10  = data_container[4].loc['F2',:], data_container[4].loc['Auc',:], data_container[4].loc['G-mean',:], data_container[4].loc['J',:]
f2_12, auc_12, g_12, j_12  = data_container[5].loc['F2',:], data_container[5].loc['Auc',:], data_container[5].loc['G-mean',:], data_container[5].loc['J',:]
f2_14, auc_14, g_14, j_14  = data_container[6].loc['F2',:], data_container[6].loc['Auc',:], data_container[6].loc['G-mean',:], data_container[6].loc['J',:]
f2_16, auc_16, g_16, j_16  = data_container[7].loc['F2',:], data_container[7].loc['Auc',:], data_container[7].loc['G-mean',:], data_container[7].loc['J',:]


f2 = np.row_stack((f2_2,f2_4,f2_6,f2_8,f2_10,f2_12,f2_14,f2_16)).T
auc = np.row_stack((auc_2,auc_4,auc_6,auc_8,auc_10,auc_12,auc_14,auc_16)).T
gm = np.row_stack((g_2,g_4,g_6,g_8,g_10,g_12,g_14,g_16)).T
j_index = np.row_stack((j_2,j_4,j_6,j_8,j_10,j_12,j_14,j_16)).T


print('F2-score:')
print(friedman_test(f2))
print('-----------------')
print('AUC')
print(friedman_test(auc))
print('-----------------')
print('G-mean')
print(friedman_test(gm))
print('-----------------')
print('J - index')
print(friedman_test(j_index))
print('-----------------')
