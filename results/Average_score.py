# -*- coding: utf-8 -*-
'''
    对标签预测正确的样本的预测分数值求平均
'''

import numpy as np
import scipy.io as scio
from tensorflow.keras import layers, optimizers, regularizers, models, metrics, losses

name_list = ['NYU_alff_ds', 'PU_alff_ds']
len_list = [212, 172]
ind = 0
path = './image/'

score_ = len_list[ind]*[0]
count = len_list[ind]*[0]
for j in range(6):
    res = scio.loadmat(path+name_list[ind]+'/' + 'lab_score_{}'.format(j) + '.mat')
    real_lab = res['real_lab']
    pred_lab = res['pred_lab']
    real_score = res['real_score']
    BHT_score = res['pred_score']
    for i in range(real_lab.shape[1]):
        if real_lab[0][i] == pred_lab[0][i]:
            score_[i] = score_[i] + BHT_score[0][i]
            count[i] = count[i] + 1
for i in range(len_list[ind]):
    score_[i] = score_[i] / count[i]
print(score_)
train_mse = metrics.MeanSquaredError()
print(train_mse(np.reshape(real_score, (-1,)), score_))

