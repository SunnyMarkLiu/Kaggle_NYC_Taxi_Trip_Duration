#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-8 上午10:12
"""
import os
import sys

import time

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import callback

# my own module
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

    shuffled_index = np.arange(0, train.shape[0])
    np.random.shuffle(shuffled_index)

    random_indexs = shuffled_index[:int(train.shape[0] * 0.70)]
    # random_indexs = np.arange(0, train.shape[0], 2)
    train = train.iloc[random_indexs, :]

    train['trip_duration'] = np.log(train['trip_duration'])
    y_train_all = train['trip_duration']
    del train['id']
    del train['trip_duration']
    id_test = test['id']
    del test['id']

    print 'train:', train.shape, ', test:', test.shape

    print 'feature check before modeling...'
    # feature_util.feature_check_before_modeling(train, test, train.columns)

    X_test = test
    df_columns = train.columns.values
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        # 'eta': 0.005,

        'min_child_weight': 1,
        'reg_lambda': 0.006,
        'reg_alpha': 0.0095,
        'scale_pos_weight': 1,
        'colsample_bytree': 1,
        'subsample': 0.93,
        'gamma': 0,
        'max_depth': 12,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'updater': 'grow_gpu',
        'gpu_id': 1,
        'nthread': -1,
        'silent': 1
    }

    learning_rates = [0.1] * 500 + [0.01] * 500 + [0.006] * 500 + [0.004] * 500 + [0.002] * 1000 + [0.001] * 1000
    dtrain = xgb.DMatrix(train, y_train_all, feature_names=df_columns)

    cv_result = xgb.cv(dict(xgb_params),
                       dtrain,
                       num_boost_round=4000,
                       early_stopping_rounds=300,
                       verbose_eval=50,
                       show_stdv=False,
                       callbacks=[callback.reset_learning_rate(learning_rates)]
                       )

    best_num_boost_rounds = len(cv_result)
    print 'best_num_boost_rounds =', best_num_boost_rounds
    # train model
    print 'training on total training data...'
    train_learning_rates = learning_rates[:best_num_boost_rounds]
    model = xgb.train(dict(xgb_params), dtrain,
                      num_boost_round=best_num_boost_rounds,
                      callbacks=[callback.reset_learning_rate(train_learning_rates)])

    print 'predict submit...'
    y_pred = model.predict(dtest)
    y_pred = np.exp(y_pred)
    df_sub = pd.DataFrame({'id': id_test, 'trip_duration': y_pred})
    submission_path = '../result/{}_submission_{}.csv.gz'.format('xgboost',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    df_sub.to_csv(submission_path, index=False, compression='gzip')


if __name__ == '__main__':
    print '========== apply xgboost model =========='
    main()
