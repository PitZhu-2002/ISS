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
cluster = '6'
kernel = ['rbf','linear','poly','sigmoid']
C = ['0.1','1','10']
metrics = ['F2','Auc','G-mean','J']
filepath = '../../parameter/setting_result/' + classifier + '/' + timeframe + '/'
root_container1 = [filepath + str(i) + '/' + C[0] + '_' + cluster + '.xlsx' for i in kernel]
root_container2 = [filepath + str(i) + '/' + C[1] + '_' + cluster + '.xlsx' for i in kernel]
root_container3 = [filepath + str(i) + '/' + C[2] + '_' + cluster + '.xlsx' for i in kernel]
#
print(root_container1)
print(root_container2)
print(root_container3)
#
data_container1 = [pd.read_excel(path).set_index('Unnamed: 0') for path in root_container1]
data_container2 = [pd.read_excel(path).set_index('Unnamed: 0') for path in root_container2]
data_container3 = [pd.read_excel(path).set_index('Unnamed: 0') for path in root_container3]

# C = 0.1
f2_01, auc_01, g_01, j_01  = data_container1[0].loc['F2',:], data_container1[0].loc['Auc',:], data_container1[0].loc['G-mean',:], data_container1[0].loc['J',:]
f2_02, auc_02, g_02, j_02  = data_container1[1].loc['F2',:], data_container1[1].loc['Auc',:], data_container1[1].loc['G-mean',:], data_container1[1].loc['J',:]
f2_03, auc_03, g_03, j_03  = data_container1[2].loc['F2',:], data_container1[2].loc['Auc',:], data_container1[2].loc['G-mean',:], data_container1[2].loc['J',:]
f2_04, auc_04, g_04, j_04  = data_container1[3].loc['F2',:], data_container1[3].loc['Auc',:], data_container1[3].loc['G-mean',:], data_container1[3].loc['J',:]

# C = 1
f2_11, auc_11, g_11, j_11  = data_container2[0].loc['F2',:], data_container2[0].loc['Auc',:], data_container2[0].loc['G-mean',:], data_container2[0].loc['J',:]
f2_12, auc_12, g_12, j_12  = data_container2[1].loc['F2',:], data_container2[1].loc['Auc',:], data_container2[1].loc['G-mean',:], data_container2[1].loc['J',:]
f2_13, auc_13, g_13, j_13  = data_container2[2].loc['F2',:], data_container2[2].loc['Auc',:], data_container2[2].loc['G-mean',:], data_container2[2].loc['J',:]
f2_14, auc_14, g_14, j_14  = data_container2[3].loc['F2',:], data_container2[3].loc['Auc',:], data_container2[3].loc['G-mean',:], data_container2[3].loc['J',:]

# C = 10
f2_101, auc_101, g_101, j_101  = data_container3[0].loc['F2',:], data_container3[0].loc['Auc',:], data_container3[0].loc['G-mean',:], data_container3[0].loc['J',:]
f2_102, auc_102, g_102, j_102  = data_container3[1].loc['F2',:], data_container3[1].loc['Auc',:], data_container3[1].loc['G-mean',:], data_container3[1].loc['J',:]
f2_103, auc_103, g_103, j_103  = data_container3[2].loc['F2',:], data_container3[2].loc['Auc',:], data_container3[2].loc['G-mean',:], data_container3[2].loc['J',:]
f2_104, auc_104, g_104, j_104  = data_container3[3].loc['F2',:], data_container3[3].loc['Auc',:], data_container3[3].loc['G-mean',:], data_container3[3].loc['J',:]



f2 = np.row_stack((f2_01,f2_02,f2_03,f2_04,f2_11,f2_12,f2_13,f2_14,f2_101,f2_102,f2_103,f2_104)).T
auc = np.row_stack((auc_01,auc_02,auc_03,auc_04,auc_11,auc_12,auc_13,auc_14,auc_101,auc_102,auc_103,auc_104)).T
gm = np.row_stack((g_01,g_02,g_03,g_04,g_11,g_12,g_13,g_14,g_101,g_102,g_103,g_104)).T
j_index = np.row_stack((j_01,j_02,j_03,j_04,j_11,j_12,j_13,j_14,j_101,j_102,j_103,j_104)).T

print('F2-score:')
#print(friedman_test(f2))
print('-----------------')
print('AUC')
#print(friedman_test(auc))
print('-----------------')
print('G-mean')
#print(friedman_test(gm))
print('-----------------')
print('J - index')
#print(friedman_test(j_index))
print('-----------------')


