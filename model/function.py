import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score


class function:
    @staticmethod
    def estimate(y_test, y_pre, prob):
        all = []
        y_test = np.array(y_test)
        result = np.array(y_pre)
        # AUC
        a = roc_auc_score(y_test, prob)
        # Precision
        precision = precision_score(y_test.astype(int),result.astype(int),pos_label = 1)
        # Recall
        recall = recall_score(y_test.astype(int), result.astype(int),pos_label = 1)
        # F2-score
        f2 =  (5 * precision * recall) / (4 * precision + recall)
        # Specificity
        specificity = recall_score(y_test.astype(int), result.astype(int), pos_label = 0)
        # G-mean
        g = math.sqrt(recall * specificity)
        # Youden's J statistic
        j = recall + specificity - 1
        # Store the result
        all.append(f2)
        all.append(a)
        all.append(g)
        all.append(recall)
        all.append(precision)
        all.append(specificity)
        all.append(j)
        return np.array(all)

    @staticmethod
    def proba_predict_minority(cls, X_test):
        # 输入 测试集
        # 输入 的 数据集 必须 是 标签为 1 的 是 少数类样本
        # 返回: 对 每一个测试集 预测是 少数类标签 的 概率的列表
        predict_prob = []
        for sample in range(0,len(X_test)):
            predict_prob.append(cls.predict_proba([X_test[sample]])[0][1])
        return np.array(predict_prob)
