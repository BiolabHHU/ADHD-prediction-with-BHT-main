
import numpy as np


# Notice: label: 0 -> ADHD, 1 -> HC
# Here, the symbol is all opposite,
# for example, yh0_inter_AD is yh0_inter_HC indeed,
# but not impact the final result
def judge_tyb(y_h0_x, y_h1_x, h_label):

    # remove the last label element, as this element is for test data
    h_label_ = h_label[:-1]
    num_h = h_label_.sum()    # calculate the frequency of 1

    # h0
    yh0_np = np.array(y_h0_x)  # deeper feature in h0

    ind_AD = np.where(h_label_.squeeze() == 1)
    ind_AD = np.squeeze(np.array(ind_AD))
    yh0_AD = yh0_np[ind_AD, :]

    ind_HC = np.where(h_label_.squeeze() == 0)
    ind_HC = np.squeeze(np.array(ind_HC))
    yh0_HC = yh0_np[ind_HC, :]

    # inter- and intra-class distance
    yh0_AD_avg = np.mean(yh0_AD, axis=(0,))
    yh0_HC_avg = np.mean(yh0_HC, axis=(0,))
    yh0_all_avg = np.mean(yh0_np, axis=(0,))

    yh0_intra_AD = np.sum(np.power(np.linalg.norm((yh0_AD - yh0_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_HC = np.sum(np.power(np.linalg.norm((yh0_HC - yh0_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_all = yh0_intra_AD + yh0_intra_HC

    yh0_inter_AD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_AD_avg), axis=0, keepdims=True), 2))
    yh0_inter_HC = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_HC_avg), axis=0, keepdims=True), 2))
    yh0_inter_all = num_h * yh0_inter_AD + (yh0_np.shape[0] - num_h) * yh0_inter_HC

    yh0_out_class = yh0_intra_all / yh0_inter_all

    # h1
    yh1_np = np.array(y_h1_x)  # deeper feature in h1
    yh1_AD = yh1_np[ind_AD, :]
    yh1_HC = yh1_np[ind_HC, :]

    # inter- and intra-class distance
    yh1_AD_avg = np.mean(yh1_AD, axis=(0,))  # h1 ADHD均值
    yh1_HC_avg = np.mean(yh1_HC, axis=(0,))  # h1 HC均值
    yh1_all_avg = np.mean(yh1_np, axis=(0,))  # 总均值

    yh1_intra_AD = np.sum(np.power(np.linalg.norm((yh1_AD - yh1_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_HC = np.sum(np.power(np.linalg.norm((yh1_HC - yh1_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_all = yh1_intra_AD + yh1_intra_HC

    yh1_inter_AD = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_AD_avg), axis=0, keepdims=True), 2))
    yh1_inter_HC = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_HC_avg), axis=0, keepdims=True), 2))
    yh1_inter_all = num_h * yh1_inter_AD + (yh1_np.shape[0] - num_h) * yh1_inter_HC

    yh1_out_class = yh1_intra_all / yh1_inter_all

    # ADHD decision function
    if yh1_out_class >= yh0_out_class:
        return True
    else:
        return False