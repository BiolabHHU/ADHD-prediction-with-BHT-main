# standard code

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import scipy.io as scio
import matplotlib.pyplot as plt
from keras import optimizers,  losses,  models
import setproctitle
from sklearn.metrics import roc_auc_score

from net_building import encode, decode, classify, prediction
from data_prepare import prepare_data, get_feature_mix
from judge import judge
from judge_tyb import judge_tyb


number_of_gpu = 0
tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(gpus[0:1], 'GPU')
for gpu in gpus: # cuda版本和cudnn版本不对应，找不到动态库链接文件
    tf.config.experimental.set_memory_growth(gpu, True) # wll2024.6.8 修改Cuda安装目录下bin目录中cusolver64_10.dll名字为cusolver64_11.dll
print(len(gpus))
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus))


loss_object = tf.keras.losses.MeanSquaredError()
loss_object2 = tf.keras.losses.SparseCategoricalCrossentropy()
loss_object3 = losses.Huber(delta=5.0)     #改成5
# optimizer = tf.keras.optimizers.legacy.Adam()  2024.6.8 wll 新环境下报错module 'keras.api._v2.keras.optimizers' has no attribute 'legacy'
optimizer = tf.keras.optimizers.Adam()
optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
setproctitle.setproctitle("wll")

#====================================================
#             training strategy
#====================================================
# epoch <50: output high_level feature of training data
def train_step_init(images, labels):
    # In prepare_data, test_data is the last element of train_data (a.k.a., images)
    # epoch < 50: traditional BHT
    # learn encoder/decoder/classifier

    with tf.GradientTape() as encode_tape, tf.GradientTape() as decode_tape, tf.GradientTape() as classify_tape:
        y = encoder(images)   # high_level features
        z = decoder(y)
        labels_s = classifier(y)  #spv: supervised

        loss1_s1 = loss_object(images, z)
        loss2_s1 = loss_object2(labels, labels_s)
        loss_sum_s1 = loss1_s1 + loss2_s1

    gradient_e = encode_tape.gradient(loss_sum_s1, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))

    gradient_d = decode_tape.gradient(loss1_s1, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, decoder.trainable_variables))

    gradient_c = classify_tape.gradient(loss2_s1, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))

    return labels_s, loss1_s1, loss2_s1   # attached symbol s1: stage 1
    # in fact, loss1_s1, loss2_s1 are the previous loss for the form turn


def train_step_init_mini_batch(images, labels, epoch):
    # In prepare_data, test_data is the last element of train_data (a.k.a., images)
    # epoch < 50: traditional BHT
    # learn encoder/decoder/classifier
    # include the test subject

    data_mini_ = tf.data.Dataset.from_tensor_slices((images, labels))
    data_mini_ = data_mini_.shuffle(buffer_size=128).batch(24)

    if epoch < 50:
        for step, (images_batch, label_batch) in enumerate(data_mini_):
            with tf.GradientTape() as encode_tape:
                y_batch = encoder(images_batch)  # high_level features
                z_batch = decoder(y_batch)
                labels_ = classifier(y_batch)  # spv: supervised

                loss1_s1 = loss_object(images_batch, z_batch)
                loss2_s1 = loss_object2(label_batch, labels_)
                loss_sum_s1 = loss1_s1 + loss2_s1

            gradient_e = encode_tape.gradient(loss_sum_s1, encoder.trainable_variables)
            optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))

    with tf.GradientTape() as encode_tape, tf.GradientTape() as decode_tape, tf.GradientTape() as classify_tape:
        y = encoder(images)  # high_level features
        z = decoder(y)
        labels_s = classifier(y)  # spv: supervised

        loss1_s1 = loss_object(images, z)
        loss2_s1 = loss_object2(labels, labels_s)
        loss_sum_s1 = loss1_s1 + loss2_s1

    gradient_e = encode_tape.gradient(loss_sum_s1, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))

    gradient_d = decode_tape.gradient(loss1_s1, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, decoder.trainable_variables))

    gradient_c = classify_tape.gradient(loss2_s1, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))

    return labels_s, loss1_s1, loss2_s1   # attached symbol s1: stage 1
    # in fact, loss1_s1, loss2_s1 are the previous loss for the form turn


# epoch >=50: score estimation
# def train_step_1(images, labels, score, BHT_label):
#     # learn encoder/predictor
#     # output estimated scores of two branches
#
#     labels_ = labels[0:-1]
#     with tf.GradientTape() as encode_tape, \
#             tf.GradientTape() as predict_AD_tape, tf.GradientTape() as predict_HC_tape,\
#             tf.GradientTape() as decode_tape, tf.GradientTape() as classify_tape:
#         y = encoder(images)  # high_level features
#         z = decoder(y)
#         y_ = y[0:-1, :]
#         labels_s = classifier(y_)
#
#         loss1_s2 = loss_object(images, z)
#         loss2_s2 = loss_object2(labels_, labels_s)
#         loss_sum_s2 = loss1_s2 + loss2_s2
#
#         score_ = score[0:-1]
#         feature_ = tf.concat((y_, images[0:-1, :]), axis=1)
#
#         feature_mix_, BHT_mid_score, ind = get_feature_mix(
#                 BHT_label, labels_, score_, feature_)
#
#         if BHT_label==1:
#             score_pred = predictor_HC(feature_mix_)
#         else:
#             score_pred = predictor_AD(feature_mix_)
#
#         loss3_s2 = 100*loss_object3(BHT_mid_score, score_pred)  #Huber loss
#         loss_final_s2 = loss3_s2+loss_sum_s2
#
#     # need save model first
#     encoder.save_weights("model_encoder.ckpt")
#
#     gradient_e = encode_tape.gradient(loss_final_s2, encoder.trainable_variables)
#     optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))
#
#     gradient_d = decode_tape.gradient(loss1_s2, decoder.trainable_variables)
#     optimizer.apply_gradients(zip(gradient_d, decoder.trainable_variables))
#
#     gradient_c = classify_tape.gradient(loss2_s2, classifier.trainable_variables)
#     optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))
#
#     if BHT_label == 1:
#         gradient_p = predict_HC_tape.gradient(loss3_s2, predictor_HC.trainable_variables)
#         optimizer1.apply_gradients(zip(gradient_p, predictor_HC.trainable_variables))
#     else:
#         gradient_p = predict_AD_tape.gradient(loss3_s2, predictor_AD.trainable_variables)
#         optimizer1.apply_gradients(zip(gradient_p, predictor_AD.trainable_variables))
#
#     y = encoder(images)  # high_level features
#     z = decoder(y)
#     y_ = y[0:-1, :]
#     labels_s = classifier(y_)
#
#     loss1_s2_new = loss_object(images, z)
#     loss2_s2_new = loss_object2(labels_, labels_s)
#     loss_sum_s2_new = loss1_s2_new + loss2_s2_new
#
#     if loss_sum_s2_new > loss_sum_s2:
#         encoder.load_weights("model_encoder.ckpt")
#
#     return labels_s, score_pred, BHT_mid_score, loss1_s2, loss2_s2, loss3_s2, ind
#     # in fact, loss1_s2, loss2_s2, loss3_s2 are the previous loss for the form turn


#=====================================================
# training model
# BHT_label = test_H0_label-1 used to select the branch for predictor
def train_h(train_data,  train_label, train_score,  print_information=False):
    #      train_h0_data, test_h0_data, train_h0_label, train_h0_score, test_h0_label
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_mse = tf.keras.losses.MeanSquaredError()

    train_x = tf.reshape(train_data, (Batch_size, train_data.shape[1]))
    label_x = np.reshape(train_label, (Batch_size,))
    score_x = np.reshape(train_score, (Batch_size,))
    (loss3, mse_, score_pred) = (-1, -1, -1)

    for epoch_count in range(EPOCH):
        if epoch_count < 20:
            optimizer.learning_rate = 0.001
        elif epoch_count < 100  and epoch_count >= 20:
            optimizer.learning_rate = 0.001
        else:
            optimizer.learning_rate = 0.001

        labels_s, loss1, loss2 = train_step_init(
            images=tf.cast(train_x, tf.dtypes.float32),
            labels=label_x)

        # labels_s, loss1, loss2 = train_step_init_mini_batch(
        #         images=tf.cast(train_x, tf.dtypes.float32),
        #         labels=label_x,
        #         epoch=epoch_count)

        train_accuracy(label_x, labels_s)

        if train_accuracy.result() > 0.9:
            break

        # print_information = True
        if print_information:
            if epoch_count % 100 == 0 or epoch_count == EPOCH-1:
                template2 = 'Epoch: {}, Loss1: {}, Loss2: {}, Loss3: {}, Mse: {}, Accuracy: {}%'
                print(template2.format(epoch_count, loss1, loss2,loss3,mse_, train_accuracy.result() * 100))

    train_data_ = train_data[0:-1, :]
    y_h0_x = encoder(train_data_)
    test_data_ = np.expand_dims(train_data[-1], axis=0)
    y_test = encoder(test_data_)

    return y_h0_x, y_test, score_pred, train_label, train_accuracy.result()
#======================================================



if __name__ == '__main__':

    name_list = ['NYU_alff_ds', 'PU_alff_ds', 'KKI_alff_ds', 'NI_alff_ds']
    dict_data = {'NYU_alff_ds': 212, 'PU_alff_ds': 172, 'KKI_alff_ds': 83, 'NI_alff_ds': 48}
    EPOCH_list = {'NYU_alff_ds': 300, 'PU_alff_ds': 300, 'KKI_alff_ds': 100, 'NI_alff_ds': 100}

    for i_out in range(1, 2):   # select ADHD-200 datasets

        name_of_data = name_list[i_out]
        Batch_size = dict_data[name_of_data]
        EPOCH = EPOCH_list[name_of_data]
        N_ensemble = 5
        seed_num = 2

        LR = 1e-3    # learning rate

        for j_out in range(0, 5):


            # predictor_HC = prediction_HC()
            # predictor_AD = prediction_AD()
            np.random.seed(seed_num)

            k = 0
            (HC2HC, HC2AD, AD2AD, AD2HC) = (0, 0, 0, 0)  # input HC to judgement result AD:  HC2AD
            (pred_tyb, pred_real) = ([], [])  # for label prediction
            (score_pred, score_real) = ([], [])  # for score aggregation

            encoder = encode()
            decoder = decode()
            classifier = classify()

            for i in range(dict_data[name_of_data]):

                # all test information is the last element of train data
                # label: 0 -> ADHD, 1 -> HC
                train_h0_data, train_h0_label, train_h0_score, \
                train_h1_data, train_h1_label, train_h1_score, rank_h0 = prepare_data(index=i + 1,
                                                                                      data_name=name_of_data)
                pred_real.append(np.rint(train_h0_label[-1]))
                score_real.append(train_h0_score[-1])

                (acc_h0_, acc_h1_, judge_record_) = ([], [], [])

                tag_ = 5
                #################################################################
                while tag_ > 0:

                    # 每个假设要返回一个test_score
                    y_h0, y_h0_test, train_predicted_score_h0, train_label_h0, acc_h0 = \
                        train_h(train_h0_data, train_h0_label, train_h0_score, print_information=False)

                    # models.save_model(encoder, './results/tyb_save/h0_encoder_{}_{}'.format(name_of_data, tag_),
                    #                   save_format='h5')
                    # models.save_model(decoder, './results/tyb_save/h0_decoder_{}_{}'.format(name_of_data, tag_),
                    #                   save_format='h5')
                    # models.save_model(classifier, './results/tyb_save/h0_classifier_{}_{}'.format(name_of_data, tag_),
                    #                   save_format='h5')
                    tf.keras.backend.clear_session()

                    y_h1, y_h1_test, train_predicted_score_h1, train_label_h1, acc_h1 = \
                        train_h(train_h1_data, train_h1_label, train_h1_score, print_information=False)

                    # models.save_model(encoder, './results/tyb_save/h1_encoder_{}_{}'.format(name_of_data, tag_),
                    #                   save_format='h5')
                    # models.save_model(decoder, './results/tyb_save/h1_decoder_{}_{}'.format(name_of_data, tag_),
                    #                   save_format='h5')
                    # models.save_model(classifier, './results/tyb_save/h1_classifier_{}_{}'.format(name_of_data, tag_),
                    #                   save_format='h5')
                    tf.keras.backend.clear_session()


                    if np.max([np.array(acc_h0), np.array(acc_h1)]) < 0.90:
                        # rebuild model
                        del encoder
                        del decoder
                        del classifier

                        np.random.seed(seed_num+2)
                        encoder = encode()
                        decoder = decode()
                        classifier = classify()
                        template1 = 'failed, tag: {}'
                        print(template1.format(tag_))

                    else:
                        tag_ = tag_ - 1
                        template2 = 'succeed, tag: {}'
                        print(template2.format(tag_))

                        judge_result = judge_tyb(y_h0, y_h1, train_label_h0)
                        judge_record_.append(np.array(judge_result))
                        acc_h0_.append(np.array(acc_h0))
                        acc_h1_.append(np.array(acc_h1))

                #################################################################





                ############################################
                # judge_result = judge_tyb(y_h0, y_h1, train_label_h0)

                if np.sum(judge_record_) >= 3:
                    judge_result = True
                    local_ = np.argmax(acc_h0_, axis=0)
                    # encoder = tf.keras.models.load_model(
                    #     './results/tyb_save/h0_encoder_{}_{}'.format(name_of_data, local_ + 1))
                    # decoder = tf.keras.models.load_model(
                    #     './results/tyb_save/h0_decoder_{}_{}'.format(name_of_data, local_ + 1))
                    # classifier = tf.keras.models.load_model(
                    #     './results/tyb_save/h0_classifier_{}_{}'.format(name_of_data, local_+1))

                else:
                    judge_result = False
                    local_ = np.argmax(acc_h1_, axis=0)
                    # encoder = tf.keras.models.load_model(
                    #     './results/tyb_save/h1_encoder_{}_{}'.format(name_of_data, local_ + 1))
                    # decoder = tf.keras.models.load_model(
                    #     './results/tyb_save/h1_decoder_{}_{}'.format(name_of_data, local_ + 1))
                    # classifier = tf.keras.models.load_model(
                    #     './results/tyb_save/h1_classifier_{}_{}'.format(name_of_data, local_+1))



                test_label_ = train_label_h0[-1]
                # label: 0 -> ADHD, 1 -> HC
                if judge_result:
                    k += 1
                if judge_result == True and test_label_ == 1:
                    HC2HC += 1
                    pred_tyb.append(1)  # category result
                    # score_pred.append(test_pred_score_h0)   # score result
                if judge_result == True and test_label_ == 0:
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

                print('\n current loop:' + str(i + 1) + ' / ' + str(dict_data[name_of_data]) + '-------------')
                print('-------------' + str(j_out + 1) + ' / ' + '50' + '-------------\n')
                print('current accuracy: ' + str(k) + '/' + str(i + 1))

            tyb1 = 'AD2AD: {}, AD2HC: {}, HC2HC: {}, HC2AD: {}'
            print(tyb1.format(AD2AD, AD2HC, HC2HC, HC2AD))
            tyb2 = '1 Accuracy: {}%'
            print(tyb2.format(100 * k / dict_data[name_of_data]))
            tyb3 = '2 Sensitivity: {}%'
            sensitivity = AD2AD / (AD2AD + AD2HC)
            print(tyb3.format(100 * AD2AD / (AD2AD + AD2HC)))
            tyb4 = '3 Specificity: {}%'
            print(tyb4.format(100 * HC2HC / (HC2AD + HC2HC)))
            tyb5 = '4 Precision: {}%'
            precision = AD2AD / (AD2AD + HC2AD)
            print(tyb5.format(100 * AD2AD / (AD2AD + HC2AD)))
            tyb6 = '5 F1 score: {}%'
            print(tyb6.format(100 * 2 * sensitivity * precision / (sensitivity + precision)))
            AUC = roc_auc_score(pred_real, pred_tyb)
            print('6 AUC:{}'.format(AUC))
            print(name_of_data)

            results_txt = str(AD2AD) + '\t' + str(AD2HC) + '\t' + str(HC2HC) + '\t' + str(HC2AD) + '\t' + str(
                100 * k / dict_data[name_of_data]) + '\t' + str(100 * AD2AD / (AD2AD + AD2HC)) + '\t' + str(
                100 * HC2HC / (HC2AD + HC2HC)) + '\t' + str(100 * AD2AD / (AD2AD + HC2AD)) + '\t' + str(
                100 * 2 * sensitivity * precision / (sensitivity + precision)) + '\t' + str(AUC) + '\n'
            with open('./results/tyb_save/'.format(name_of_data) + name_of_data + '.txt',
                      "a+") as f:
                f.write(results_txt)






