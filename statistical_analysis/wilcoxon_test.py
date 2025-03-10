# -*- coding: utf-8 -*-
import os
import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy.stats import wilcoxon
np.set_printoptions(suppress=True)
def wilcoxon_test(my_path,comp_path):
    ours = pd.read_excel(my_path).set_index('Unnamed: 0')
    comp = pd.read_excel(comp_path).set_index('Unnamed: 0')
    our_r2, com_r2 = ours.iloc[1,:100], comp.iloc[1, :100]
    our_auc, com_auc = ours.iloc[2,:100], comp.iloc[2,:100]
    our_G, com_G = ours.iloc[3,:100], comp.iloc[3,:100]
    our_f, com_f = ours.iloc[4,:100], comp.iloc[4,:100]
    print('F2')
    f2  = wilcoxon(our_r2, com_r2)
    print('AUC')
    auc = wilcoxon(our_auc,com_auc)
    print('G-mean')
    g = wilcoxon(our_G,com_G)
    print('J')
    f = wilcoxon(our_f, com_f)





timeframe = 't-3'
classifier = 'mlp'
comp_fpath = '../result/' + timeframe + '/' + classifier
parameter = '0.1_6'
my_path = '../parameter/setting_result/' + classifier + '/' + timeframe + '/rbf/' + parameter + '.xlsx'
file_name = os.listdir(comp_fpath)
df = pd.DataFrame(columns = ['AUC', 'RA+', 'RA-', 'G-mean', 'RG+', 'RG-', 'F2', 'RF2+', 'RF2-', 'F1', 'RF1+', 'RF1-','J','RJ+','RJ-'])
for setting in file_name:
    if setting != 'ISSR.xlsx':
        print('Ours VS', setting[:-4])
        li = wilcoxon_test(comp_path = comp_fpath + '/' + setting, my_path = my_path)
        df.loc[str(setting)] = li
        print('-------------------')



