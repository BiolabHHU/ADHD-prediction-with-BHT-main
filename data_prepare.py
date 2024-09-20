from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()


#====================================================
#             data preprocess
#====================================================

# In prepare_data, test_data is the last element of train_data (a.k.a., images)
def prepare_data(index, data_name):

    train_h0_data, train_h0_label, train_h0_score,\
    train_h1_data, train_h1_label,  train_h1_score, rank_h0\
     = eng.svm_two_suppose_ALFF_tyb1(index, data_name, nargout=7)

    train_h0_data  = np.array(train_h0_data)
    train_h0_label = np.array(train_h0_label)
    train_h0_score = np.array(train_h0_score)

    train_h1_data  = np.array(train_h1_data)
    train_h1_label = np.array(train_h1_label)
    train_h1_score = np.array(train_h1_score)

    # normalized score for ADHD and HC, respectively
    # [train_h0_score, bound] = eng.score_normalize(index, data_name, nargout=2)
    # train_h0_score = np.array(train_h0_score)
    # train_h1_score = train_h0_score

    # test_h0_data  = train_h0_data[-1,:]
    # test_h0_label = train_h0_label[-1,:]
    # test_h0_score = train_h0_score[-1,:]
    #
    # test_h1_data = train_h1_data[-1, :]
    # test_h1_label = train_h1_label[-1, :]
    # test_h1_score = train_h1_score[-1, :]

    # num_h0 = train_h0_label.sum()  # ADHD subjects in h0 hypothesis
    # num_h1 = train_h1_label.sum()  # ADHD subjects in h1 hypothesis


    # train_h0_data train_h1_data：     normalized data
    # test_h0_data test_h1_data：       normalized data
    # Notice: train_data attached with test_data as the last elements
    # train_h0_label train_h1_label：   1->ADHD  0-HC
    # test_h0_label test_h1_label：     1->ADHD  0-HC

    return train_h0_data, train_h0_label, train_h0_score,\
           train_h1_data, train_h1_label, train_h1_score, rank_h0#, bound
           # test_h0_data,  test_h0_label,  test_h0_score,\
           # test_h1_data,  test_h1_label,  test_h1_score,\
           # num_h0, num_h1

# Here, label_branch in fact is the ground-truth label of test data
def get_feature_mix(label_branch, label, score, feature):
    # 取出两类样本的分界线
    ind_ = np.where(label == label_branch)
    length = len(ind_[0])

    feature_ = np.array(feature)
    feature_ = feature_[ind_]
    score_ = score[ind_]

    # if label_branch == 1:
    #     # feature = feature[:length]
    #     # score = score[:length]
    #     # ind = np.where(label == True)
    # else:
    #     feature = feature[-length:]
    #     score = score[-length:]
    #     ind = np.where(label == False)
    return feature_, score_, ind_[0]
#====================================================
