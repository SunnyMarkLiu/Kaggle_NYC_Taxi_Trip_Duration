#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-10 下午4:52
"""
import os
import sys

import time

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import lightgbm as lgbm

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

    random_indexs = np.arange(0, train.shape[0], 1)
    train = train.iloc[random_indexs, :]

    train['trip_duration'] = np.log(train['trip_duration'])
    y_train_all = train['trip_duration']
    # del train['id']
    del train['trip_duration']
    id_test = test['id']
    # del test['id']

    train['id'] = train['id'].map(lambda i: int(i[2:]))
    test['id'] = test['id'].map(lambda i: int(i[2:]))

    print 'train:', train.shape, ', test:', test.shape

    print 'feature check before modeling...'
    # feature_util.feature_check_before_modeling(train, test, train.columns)

    d_train = lgbm.Dataset(train, label=y_train_all)

    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'num_leaves': 2 ** 4,
        'learning_rate': 0.05,
        'max_depth': -1,
        'max_bin': 255,
        'subsample_for_bin': 50000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_alpha': 1,
        'reg_lambda': 0,
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight': 1,
        'early_stopping_round': 20,
        'metric': 'rmsle',
        'verbose': 0
    }

    def lgb_rmsle_score(preds, dtrain):
        labels = np.exp(dtrain.get_label())
        preds = np.exp(preds.clip(min=0))
        return 'rmsle', np.sqrt(np.mean(np.square(np.log1p(preds) - np.log1p(labels)))), False

    cv_results = lgbm.cv(lgbm_params,
                         d_train,
                         num_boost_round=5000,
                         nfold=3,
                         feval=lgb_rmsle_score,
                         early_stopping_rounds=300,
                         verbose_eval=50)

    best_num_boost_rounds = len(cv_results['rmsle-mean'])
    print 'best_num_boost_rounds =', best_num_boost_rounds
    # train model
    print 'training on total training data...'
    model = lgbm.train(lgbm_params, d_train, num_boost_round=best_num_boost_rounds)

    print 'predict submit...'
    y_pred = model.predict(test)
    y_pred = np.exp(y_pred)
    df_sub = pd.DataFrame({'id': id_test, 'trip_duration': y_pred})
    submission_path = '../result/{}_submission_{}.csv.gz'.format('lightgbm',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    df_sub.to_csv(submission_path, index=False, compression='gzip')


if __name__ == '__main__':
    print '========== apply lightgbm model =========='
    main()
