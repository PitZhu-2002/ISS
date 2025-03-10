import os
import numpy as np
import pandas as pd


def statistic(number_list,timeframe,classifier,fpath):
    df1 = pd.DataFrame(
        index = ['F2','Auc', 'G-mean', 'Recall', 'Precision','Specificity','J']
    )
    df2 = pd.DataFrame(
        index = ['F2','Auc', 'G-mean', 'Recall', 'Precision','Specificity','J']
    )
    for i in number_list:
        fname = fpath + '/' + str(i) + '.xlsx'
        ours = pd.read_excel(fname).set_index('Unnamed: 0')
        # 每个指标统计数据
        mean_list = [np.average(ours.iloc[i, :ours.shape[1]]) for i in range(1,ours.shape[0])]
        std_list = [np.std(ours.iloc[i, :ours.shape[1]]) for i in range(1,ours.shape[0])]
        df1[i] = mean_list
        df2[i] = std_list
    df1.to_excel('./' + classifier + '/mean/' + timeframe + '/' + 'mean_' + classifier + '.xlsx')
    df2.to_excel('./' + classifier + '/std/' + timeframe + '/' + 'std_' + classifier + '.xlsx')


timeframe = 't-3'
classifier = 'tree'
fpath = '../result/' + timeframe + '/' + classifier
file_name = os.listdir(fpath)
names = []
for i in file_name:
    if i[-5:] == '.xlsx':
        names.append(i[:-5])
statistic(names,timeframe = timeframe,classifier = classifier,fpath = fpath)
