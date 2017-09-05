#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-8 上午10:12
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# my own module
from conf.configure import Configure
from utils import data_utils, feature_util


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

    random_indexs = np.arange(0, train.shape[0], 10)
    train = train.iloc[random_indexs, :]

    train['trip_duration'] = np.log(train['trip_duration'])
    y_train_all = train['trip_duration']
    del train['id']
    del train['trip_duration']
    id_test = test['id']
    del test['id']

    print 'train:', train.shape, ', test:', test.shape

    print 'feature check before modeling...'
    feature_util.feature_check_before_modeling(train, test, train.columns)

    train_rmses = []
    val_rmses = []
    num_boost_roundses = []

    X_test = test
    df_columns = train.columns.values
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.005,
        'max_depth': 4,
        'subsample': 0.93,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'updater': 'grow_gpu',
        'gpu_id': 1,
        'silent': 1
    }

    for i in range(0, 1):
        random_state = 42 + i
        X_train, X_val, y_train, y_val = train_test_split(train, y_train_all, test_size=0.25, random_state=random_state)
        print 'X_train:', X_train.shape, ', X_val:', X_val.shape

        dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
        dval = xgb.DMatrix(X_val, y_val, feature_names=df_columns)

        cv_result = xgb.cv(dict(xgb_params),
                           dtrain,
                           num_boost_round=1000,
                           early_stopping_rounds=50,
                           verbose_eval=50,
                           show_stdv=False
                           )

        num_boost_rounds = len(cv_result)
        num_boost_roundses.append(num_boost_rounds)
        model = xgb.train(dict(xgb_params), dtrain, num_boost_round=num_boost_rounds)
        train_rmse = mean_squared_error(dtrain.get_label(), model.predict(dtrain))
        val_rmse = mean_squared_error(dval.get_label(), model.predict(dval))
        print 'perform {} cross-validate: train rmse = {}, validate rmse = {}'.format(i + 1, train_rmse,
                                                                                      val_rmse)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

    print '\naverage train rmse = {} average validate rmse = {}'.format(
        sum(train_rmses) / len(train_rmses),
        sum(val_rmses) / len(val_rmses))

    best_num_boost_rounds = sum(num_boost_roundses) // len(num_boost_roundses)
    print 'best_num_boost_rounds =', best_num_boost_rounds
    # train model
    print 'training on total training data...'
    dtrain_all = xgb.DMatrix(train, y_train_all, feature_names=df_columns)
    model = xgb.train(dict(xgb_params, base_score=np.mean(y_train_all)), dtrain_all,
                      num_boost_round=best_num_boost_rounds)

    print 'predict submit...'
    y_pred = model.predict(dtest)
    y_pred = np.exp(y_pred)
    df_sub = pd.DataFrame({'id': id_test, 'trip_duration': y_pred})
    df_sub.to_csv(Configure.submission_path, index=False)


if __name__ == '__main__':
    print '========== apply xgboost model =========='
    main()
