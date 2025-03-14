import math
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

class ISSR:
    def __init__(self,data,bootstrap = True,cluster = 3,random_state = 10, kernel = 'rbf', c = 0.1):
        self.data = data
        if bootstrap == True:
            data = data.sample(len(data),replace=True,random_state = random_state)
        else:
            data = data
        self.random_state = random_state
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]
        # 多数类
        self.majority = data[data.iloc[:,-1] == 0 ]
        self.maj_id = np.arange(0,len(data[data.iloc[:,-1] == 0 ]))   # 表示所有数据的下标
        # 少数类
        self.minority = data[data.iloc[:,-1] == 1 ]
        # 采样得的平衡数据
        self.prime = deepcopy(self.minority)
        # 不平衡整数比例
        self.ratio = math.floor(len(self.maj_id) / len(self.minority))
        self.cluster = cluster
        self.kernel = kernel
        self.c = c

    def split(self,n, data):
        '''
        :param n: 将 data 划分成 n 份
        :param data: 划分数据
        :return: 划分的结果
        举例： 传入 n = 8,data 大小为 1290
        返回: [162,162,161,161,161,161,161,161]
        '''
        size = len(data)
        evy = math.floor(size / n)
        minus = size - n * evy
        base = np.array([evy for i in range(0, n)])
        base[:minus] = base[:minus] + 1
        return base

    def disk(self,cluster, spt, compile, judge):
        '''
        :param cluster: 某一个簇的 下标集合
        :param spt: self.split() 的结果
        :param compile:
        :param judge:
        :return:
        '''
        a = 0
        for idx,i in enumerate(spt):
            if judge == 0:
                compile.append(cluster[a: a + i]) # [0-3)
            # elif judge == -1:
            #     compile[idx].extend(cluster[a:])    # [5,7]
            else:
                compile[idx].extend(cluster[a:a + i])   # [3,5)
            a = a + i
        return compile

    def average_cluster(self,data,rate):
        # 传入的 data 是  [[0 1 5],[2 4],[3 6]]
        compile = []    # 保存划分 cluster 的结果
        for idx, cluster in enumerate(data):
            # cluster 表示每个簇中的下标
            spt = self.split(rate, cluster)   # ratio = 3 cluster = [2,3,4,5,1,0,6] -> spt = [3 2 2]
            if idx == 0:
                compile = self.disk(cluster, spt, compile, judge=0)  # 分好的每类
            # elif idx == len(data) - 1:
            #     compile = self.disk(cluster, spt, compile, judge=-1)
            else:
                compile = self.disk(cluster, spt, compile, judge=1)
        return compile

    def equal_split(self, csf):
        rate = math.floor(len(self.concat(csf)) / len(self.minority))
        #rate = math.ceil(len(self.concat(csf)) / len(self.minority))
        idx_set = self.average_cluster(csf,rate)  # 平均划分后的数据
        return idx_set,rate
    def classify(self,cluster,maj_idx):
        # 作用: 将同一类别的 maj_idx 集合在一起
        # cluster 表示 最终聚类的所有结果，如[0 1 1 0 2 2 2 1 1 0 2]
        label = list(set(cluster))
        csf = [[] for i in range(len(label))]
        for i in range(len(cluster)):
            lb = cluster[i]
            csf[lb].append(maj_idx[i])
        # 返回各簇在原数据的地址
        # 比如 maj_idx 为 [0 1 2 3 4 5 6]
        # cluster 为 [0 0 1 2 1 0 2]
        # csf 为 [[0 1 5],[2 4],[3 6]]
        return csf
    def KMeans_classify(self,maj_idx):
        # 传入的 data 要求是要进行聚类的 maj 数据
        X = self.majority.iloc[maj_idx].iloc[:,:-1] # maj_idx下的 feature数据
        kmeans = KMeans(n_clusters = self.cluster,random_state = self.random_state).fit(X)   # 聚类
        # 传入: kmeans.labels_ 表示 上面聚类出的结果，例如:[0 1 1 0 2 2 2 1 1 0 2]
        # 传入: maj_idx 剩余的多数类的地址
        csf = self.classify(kmeans.labels_,maj_idx) # 在原数据的地址
        # 返回各簇在原数据的地址
        # 比如 maj_idx 为 [0 1 2 3 4 5 6]
        # cluster 为 [0 0 1 2 1 0 2]
        # csf 为 [[0 1 5],[2 4],[3 6]]
        return csf
    def support_vector(self, csf):
        # data: (1:1) 数据集 D 最后一列是 label
        # dif 是 已有用 sv 组成的少数类 和 多数类的差值
        # judge : 数量是否有 majority多
        # size: 差值
        for d in range(len(csf)):
            csf[d] = shuffle(csf[d],random_state = self.random_state)
        sup_compile = []  # 存储 support vector
        term = 0
        eq_sp,rate = self.equal_split(csf)   # 返回各个簇都找一些的数据 是下标
        for i in range(rate): # self.ratio可能要调整 重新计算一下不平衡比例
            idx = np.array(eq_sp[i])
            data = pd.concat([self.majority.iloc[idx], self.minority], axis=0)
            svm = SVC(kernel = self.kernel, C = self.c, probability = True, random_state = self.random_state)
            svm.fit(data.iloc[:, :-1], data.iloc[:, -1])
            first_idx = idx[svm.support_[data.iloc[svm.support_]['label'] == 0]]  # 原样本中找出的 spv 的位置
            first_support_vector = deepcopy(self.majority).iloc[first_idx,:-1]
            spv_idx =  first_idx[svm.predict(first_support_vector) == 0]
            term = term + 1
            sup_compile.extend(spv_idx)  # 存储到 sup_compile 里面
        for i in range(len((csf))):
            csf[i] = shuffle(list(set(csf[i]) - set(sup_compile)),random_state = self.random_state)
        return sup_compile
    def concat(self,data):
        back = []
        for d in data:
            back.extend(d)
        return np.array(back)
    def generate(self):
        csf = deepcopy(self.maj_id)     # 复制 self.majority 的顺序
        mark = 0                        # 记录已过滤出的 Support Vector 的个数
        count = 1                       # 记录轮次的 可以删除
        select_maj = []                 # 存储 过滤出的 Support Vector 的位置
        # 问题: 按照循环聚类 末尾的问题
        rd_st = deepcopy(self.random_state)
        while mark < len(self.minority):
            csf = shuffle(csf,random_state = self.random_state)
            csf = self.KMeans_classify(csf)
            spvm = self.support_vector(csf)
            csf = self.concat(csf)
            a = np.sort(spvm)
            select_maj.extend(spvm)
            mark = mark + len(spvm)
            rd_st = rd_st + 1
            count = count + 1
        select_maj = np.array(select_maj)
        self.prime = pd.concat([self.prime,self.majority.iloc[select_maj].sample(len(self.minority),random_state = self.random_state)],axis = 0)
        self.prime.iloc[:,-1] = self.prime.iloc[:,-1].astype(int)