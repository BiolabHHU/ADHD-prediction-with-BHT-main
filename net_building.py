import numpy as np
import tensorflow as tf
import scipy.io as scio
import sys
import matplotlib.pyplot as plt
from keras import layers, optimizers, regularizers, models, metrics, losses

#==================================================
#                 Network Setting
#==================================================
num_raw_feature = 35
num_of_hidden = 20  # neural unit in auto-coding network
num_of_hidden_classify = 8  # neural unit in classification network

# encode network
def encode():
    inputs = tf.keras.layers.Input(shape=[num_raw_feature, ])      # input feature number = 50
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.ReLU()
    layer3 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')


    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    y = layer3(y)

    return tf.keras.Model(inputs=inputs, outputs=y)

# decode network
def decode():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(num_raw_feature, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.Reshape(target_shape=(num_raw_feature, 1))
    layer3 = tf.keras.layers.ReLU()

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    y = layer3(y)
    return tf.keras.Model(inputs=inputs, outputs=y)

# residual_block for classification of hidden feature
def residual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()  # 采用sequential构造法
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result

# classification network
def classify():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])

    block_stack_1 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_2 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_3 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]

    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer_in = tf.keras.layers.Dense(num_of_hidden_classify, kernel_initializer='he_normal', activation='relu')
    layer_out = tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')

    res_x_0 = 0
    res_x_1 = 0
    res_x_2 = 0

    x = inputs
    x = layer0(x)
    x = layer_in(x)

    x_0 = x
    for block in block_stack_1:
        res_x_0 = block(x)
    x = res_x_0 + x

    for block in block_stack_2:
        res_x_1 = block(x)
    x = res_x_1 + x

    for block in block_stack_3:
        res_x_2 = block(x)
    x = res_x_2 + x

    x = x_0 + x
    x = layer_out(x)   # output dimension: 2
    return tf.keras.Model(inputs=inputs, outputs=x)

def prediction():
    # inputs = layers.Input(shape=[num_of_hidden + num_raw_feature, ])
    # bn = layers.BatchNormalization(input_dim=num_of_hidden + num_raw_feature)
    inputs = layers.Input(shape=[num_raw_feature, ])
    bn = layers.BatchNormalization(input_dim=num_raw_feature)
    dn1 = layers.Dense(32, kernel_initializer='random_uniform',  # 均匀初始化
                    activation='relu',  # relu激活函数
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),  # L1及L2 正则项
                    use_bias=True)
    bn1 = layers.BatchNormalization(input_dim=32)
    dn2 = layers.Dense(16, kernel_initializer='random_uniform',  # 均匀初始化
                    activation='relu',  # relu激活函数
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),  # L1及L2 正则项
                    use_bias=True)
    bn2 = layers.BatchNormalization(input_dim=16)
    # drop = layers.Dropout(0.1)
    dn3 = layers.Dense(1,  use_bias=True)
    y = bn(inputs)
    y = dn1(y)
    y = bn1(y)
    y = dn2(y)
    y = bn2(y)
    #    y = drop(y)
    y = dn3(y)

    # if name_list_=='PU_alff_ds':
    # y = tf.clip_by_value(y, 18, 44)
    # if name_list_=='NYU_alff_ds':
    # y = tf.clip_by_value(y, 40, 62)

    return models.Model(inputs=inputs, outputs=y, name='score_prediction_net')


