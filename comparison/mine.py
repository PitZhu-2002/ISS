from copy import deepcopy
from sklearn import tree
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from Imbalance_Learning.ISSR.model import function
from Imbalance_Learning.ISSR.model.issr import ISSR
warnings.filterwarnings("ignore")


def operate(data,cluster,base_copy,random_state):
    # 基分类器初始化
    base = deepcopy(base_copy)
    svm_sam = ISSR(data = data,bootstrap = False,cluster = cluster,random_state = random_state)
    svm_sam.generate()
    base.fit(svm_sam.prime.iloc[:,:-1] , svm_sam.prime.iloc[:,-1])
    return base

def start(data,times,cluster,base_copy,base_name,timeframe,random_state):
    svmk_PM = [[] for i in range(7)]       # Support Vector K-means 5个指标
    svmk_ACC = []                   # Support Vector K-means 准确率
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    stf_split = StratifiedShuffleSplit(n_splits = times,test_size = 0.1,random_state = 10)
    id = 0
    rd_i = 0
    for train_index, test_index in stf_split.split(X, y):
        X_prime, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_prime, y_test = np.array(y)[train_index], np.array(y)[test_index]
        id = id + 1
        base_svmk = operate(data.iloc[train_index],cluster = cluster,base_copy = base_copy,random_state = random_state)
        rd_i = rd_i + 1
        back1 = function.estimate(y_test = y_test, y_pre = base_svmk.predict(X_test), prob = function.proba_predict_minority(base_svmk, X_test))
        for l in range(len(svmk_PM)):
            svmk_PM[l].append(back1[l])
        svmk_ACC.append(base_svmk.score(X_test, y_test))
    df_roc = ['F2','Auc', 'G-mean', 'Recall', 'Precision','Specificity','J']
    per2 = pd.DataFrame(columns=[i for i in range(times)])
    per2.loc['Acc'] = svmk_ACC
    for p in range(len(svmk_PM)):
        per2.loc[df_roc[p]] = svmk_PM[p]
    per2.to_excel('../result/t-' + timeframe + '/'+ base_name + '/ISSR.xlsx')

if __name__ == '__main__':
    base_dic = {
        'tree': tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=10, min_samples_leaf=2, random_state=10),
        'mlp': MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='relu', max_iter=1000,random_state = 10)
    }
    # Key parameter controlling the large-scale experiments
    name = 'tree'
    base = base_dic[name]
    timeframe = '3'
    cluster = 6
    times = 100
    random_state = 71
    #
    data = pd.read_excel('../data/data_' + timeframe + '_std.xls')
    start(data,times = times, cluster = cluster,base_copy = base,base_name = name,timeframe = timeframe,random_state = random_state)
