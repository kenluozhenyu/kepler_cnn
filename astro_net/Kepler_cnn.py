from keras.layers import Dense, Flatten, Dropout, Input
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import (Conv1D, MaxPooling1D)
from keras.layers.merge import concatenate

from configure import environment


def build_Kepler_CNN():

    # 2001/201: length of the sample
    # 1 : y coordinate

    global_view_shape = (2001, 1)
    local_view_shape = (201, 1)

    # 1st input model
    visible1 = Input(shape=global_view_shape)
    conv11 = Conv1D(16, kernel_size=5, activation='relu')(visible1)
    conv12 = Conv1D(16, kernel_size=5, activation='relu')(conv11)
    pool11 = MaxPooling1D(pool_size=5, strides=2)(conv12)
    conv13 = Conv1D(32, kernel_size=5, activation='relu')(pool11)
    conv14 = Conv1D(32, kernel_size=5, activation='relu')(conv13)
    pool12 = MaxPooling1D(pool_size=5, strides=2)(conv14)
    conv15 = Conv1D(64, kernel_size=5, activation='relu')(pool12)
    conv16 = Conv1D(64, kernel_size=5, activation='relu')(conv15)
    pool13 = MaxPooling1D(pool_size=5, strides=2)(conv16)
    conv17 = Conv1D(128, kernel_size=5, activation='relu')(pool13)
    conv18 = Conv1D(128, kernel_size=5, activation='relu')(conv17)
    pool14 = MaxPooling1D(pool_size=5, strides=2)(conv18)
    conv19 = Conv1D(256, kernel_size=5, activation='relu')(pool14)
    conv110 = Conv1D(256, kernel_size=5, activation='relu')(conv19)
    pool15 = MaxPooling1D(pool_size=5, strides=2)(conv110)
    flat1 = Flatten()(pool15)

    # 2nd input model
    visible2 = Input(shape=local_view_shape)
    conv21 = Conv1D(16, kernel_size=5, activation='relu')(visible2)
    conv22 = Conv1D(16, kernel_size=5, activation='relu')(conv21)
    pool21 = MaxPooling1D(pool_size=7, strides=2)(conv22)
    conv23 = Conv1D(32, kernel_size=5, activation='relu')(pool21)
    conv24 = Conv1D(32, kernel_size=5, activation='relu')(conv23)
    pool22 = MaxPooling1D(pool_size=7, strides=2)(conv24)
    flat2 = Flatten()(pool22)

    # merge input models
    merge = concatenate([flat1, flat2])

    # interpretation model
    hidden1 = Dense(512, activation='relu')(merge)
    hidden1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(512, activation='relu')(hidden1)
    hidden2 = Dropout(0.5)(hidden2)
    hidden3 = Dense(512, activation='relu')(hidden2)
    hidden3 = Dropout(0.5)(hidden3)
    hidden4 = Dense(512, activation='relu')(hidden3)
    hidden4 = Dropout(0.5)(hidden4)

    output = Dense(environment.NB_CLASSES, activation='softmax')(hidden4)

    model = Model(inputs=[visible1, visible2], outputs=output)

    # summarize layers
    # print(model.summary())

    return model
