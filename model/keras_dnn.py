#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-11 下午7:33
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import time
import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras import optimizers, regularizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils.keras_utils import ModelCheckpointAndLearningRateDecay

# my own module
from conf.configure import Configure
from utils import data_utils


def main():
    print 'load datas...'

    # final operate dataset
    files = os.listdir('../input')
    op_scope = 0
    for f in files:
        if 'operate' in f:
            op = int(f.split('_')[1])
            if op > op_scope:
                op_scope = op

    print 'load dataset from op_scope = {}'.format(op_scope)
    train, test = data_utils.load_dataset(op_scope)
    train.drop(['pickup_datetime', 'dropoff_datetime'], axis=1, inplace=True)
    test.drop(['pickup_datetime', 'dropoff_datetime'], axis=1, inplace=True)

    shuffled_index = np.arange(0, train.shape[0], 1)
    np.random.shuffle(shuffled_index)

    random_indexs = shuffled_index
    train = train.iloc[random_indexs, :]

    train['trip_duration'] = np.log(train['trip_duration'])
    y_train_all = train['trip_duration']
    # del train['id']
    del train['trip_duration']
    id_test = test['id']
    # del test['id']

    train['id'] = train['id'].map(lambda i: int(i[2:]))
    test['id'] = test['id'].map(lambda i: int(i[2:]))

    train.fillna(0, inplace=True)
    print 'train:', train.shape, ', test:', test.shape
    X_train, X_val, y_train, y_val = train_test_split(train.values, y_train_all.values, test_size=0.25, random_state=42)

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    input_layer = Input((train.shape[1],))
    top_model = Dense(128, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(5e-4))(input_layer)
    top_model = Dropout(0.8)(top_model)
    top_model = Dense(256, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(5e-4))(top_model)
    top_model = Dropout(0.8)(top_model)
    top_model = Dense(128, activation='relu', name='fc3', kernel_regularizer=regularizers.l2(5e-4))(top_model)
    top_model = Dropout(0.8)(top_model)
    top_model = Dense(64, activation='relu', name='fc4', kernel_regularizer=regularizers.l2(5e-4))(top_model)
    top_model = Dropout(0.8)(top_model)
    top_model = Dense(1, name='predictions')(top_model)

    model = Model(input=input_layer, output=top_model, name='dnn')

    model.compile(loss=root_mean_squared_error,
                  optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
    print(model.summary())

    weights_file = Configure.best_keras_dnn_model_weights
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    checkpoint_lr_decay = ModelCheckpointAndLearningRateDecay(weights_file,
                                                              lr_decay=0.9,
                                                              monitor='val_loss', verbose=0,
                                                              save_best_only=True, mode='min',
                                                              patience=3)

    print '========== start training =========='
    print 'training data size: ', X_train.shape[0]
    print 'validate data size: ', X_val.shape[0]
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
        print("Model loaded.")

    model.fit(X_train, y=y_train, batch_size=10000, epochs=100, verbose=1,
              validation_data=(X_val, y_val), shuffle=True, callbacks=[earlystop, checkpoint_lr_decay])
    print("Training Finished!")
    print '============ load weights ============'
    model.load_weights(weights_file)
    print '========== start validating =========='
    predict_val = model.predict(X_val, batch_size=100, verbose=1)
    print '\nvalidate rmse =', mean_squared_error(predict_val, y_val)

    print '========== start predicting =========='
    predict_test = model.predict(test.values)
    predict_test = np.exp(predict_test)
    df_sub = pd.DataFrame({'id': id_test.values, 'trip_duration': predict_test})
    submission_path = '../result/{}_submission_{}.csv.gz'.format('keras_dnn',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    df_sub.to_csv(submission_path, index=False, compression='gzip')


if __name__ == '__main__':
    print '========== apply deep neural network =========='
    main()
