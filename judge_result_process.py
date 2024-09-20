import numpy as np

def judge_result_process(train_label_h0, judge_result, k, j,
                         HC2HC,  AD2AD, HC2AD, AD2HC, pred_tyb):
    test_label_ = train_label_h0[-1]
    # label: 0 -> ADHD, 1 -> HC
    if judge_result:
        k += 1
    if judge_result == True and test_label_ == 1:
        j += 1
        HC2HC += 1
        pred_tyb.append(1)  # category result
        # score_pred.append(test_pred_score_h0)   # score result
    if judge_result == True and test_label_ == 0:
        j += 1
        AD2AD += 1
        pred_tyb.append(0)
        # score_pred.append(test_pred_score_h0)
    if judge_result == False and test_label_ == 1:
        HC2AD += 1
        pred_tyb.append(0)
        # score_pred.append(test_pred_score_h1)
    if judge_result == False and test_label_ == 0:
        AD2HC += 1
        pred_tyb.append(1)
        # score_pred.append(test_pred_score_h1)
    return k, j, HC2HC,  AD2AD, HC2AD, AD2HC, pred_tyb
