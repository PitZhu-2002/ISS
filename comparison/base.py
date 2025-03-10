from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
from sklearn.neural_network import MLPClassifier
from Imbalance_Learning.ISSR.model import function
warnings.filterwarnings("ignore")
def start(name,data,times,base_copy,base_name,timeframe):
    PM_total = []                   # 对比方法 Auc、F-measure、G-measure、Recall、Specificity 集合
    ACC_total = []                  # 对比方法准确率
    for i in range(len(name)):
        PM_total.append([[] for j in range(7)])
        ACC_total.append([])                # 每个下标代表一个 准确率
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    stf_split = StratifiedShuffleSplit(n_splits = times,test_size = 0.1,random_state = 10)
    id = 0
    for train_index, test_index in stf_split.split(X, y):
        X_prime, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_prime, y_test = np.array(y)[train_index], np.array(y)[test_index]
        print(id)
        id = id + 1
        pool_classifier = []  # 下标 表示分类器的种类，对应 name 里面过采样方法产生的数据
        pool_prd = []
        pool_ppb = []
        for m in range(len(name)):
            base = deepcopy(base_copy)
            n = name[m]
            base.fit(X_prime,y_prime)
            pool_classifier.append(base)
            pool_prd.append(base.predict(X_test))
            pool_ppb.append(function.proba_predict_minority(base, X_test))
            back = function.estimate(
                y_test = y_test,
                y_pre = base.predict(X_test),
                prob = function.proba_predict_minority(base, X_test)
            )
            for i in range(len(back)):
                PM_total[m][i].append(back[i])
            ACC_total[m].append(base.score(X_test, y_test))
    df_roc = ['F2','Auc', 'G-mean', 'Recall', 'Precision','Specificity','J']
    for type in range(len(names)):
        per = pd.DataFrame(columns=[i for i in range(times)])
        per.loc['Acc'] = ACC_total[type]
        for j in range(len(PM_total[type])):
            per.loc[df_roc[j]] = PM_total[type][j]
        per.to_excel('../result/t-' + timeframe + '/' + base_name + '/' + names[type] + '.xlsx')



if __name__ == '__main__':
    base_dic = {
        'tree': tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 10, min_samples_leaf = 2,random_state = 10),
        'mlp': MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='relu', max_iter=1000, random_state=10)
    }
    timeframe = '3'
    names = ['base']
    data = pd.read_excel('../data/data_'+ timeframe + '_std.xls')
    times = 100
    name = 'tree'
    base = base_dic[name]
    start(names,data,times = times, base_copy = base, base_name = name, timeframe = timeframe)
