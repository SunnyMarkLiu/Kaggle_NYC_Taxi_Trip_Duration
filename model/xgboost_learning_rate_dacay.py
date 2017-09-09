#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-9 下午3:13
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# remove warnings
import warnings
warnings.filterwarnings('ignore')

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

    random_indexs = np.arange(0, train.shape[0], 10)
    train = train.iloc[random_indexs, :]

    train['trip_duration'] = np.log(train['trip_duration'])
    y_train_all = train['trip_duration']
    del train['id']
    del train['trip_duration']
    id_test = test['id']
    del test['id']

    print 'train:', train.shape, ', test:', test.shape

    X_test = test
    df_columns = train.columns.values
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.01,
        'min_child_weight': 1,
        'reg_lambda': 0.006,
        'reg_alpha': 0.0095,
        'scale_pos_weight': 1,
        'colsample_bytree': 1,
        'subsample': 0.93,
        'gamma': 0,
        'max_depth': 8,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'updater': 'grow_gpu',
        'gpu_id': 0,
        'nthread': -1,
        'silent': 1
    }
    random_state = 42
    X_train, X_val, y_train, y_val = train_test_split(train, y_train_all, test_size=0.25, random_state=random_state)
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dvalid = xgb.DMatrix(X_val, y_val, feature_names=df_columns)

    min_learning_rate = 0.0001
    learning_rate = xgb_params["eta"]
    learning_rate_decay = 0.8
    best_model = None
    best_valid_rmse = 100

    while learning_rate > min_learning_rate:
        print '================ learning rate = {} ================'.format(xgb_params["eta"])
        model = xgb.train(dict(xgb_params),
                          dtrain,
                          num_boost_round=5000,
                          evals=[(dtrain, 'train'), (dvalid, 'valid')],
                          early_stopping_rounds=60,
                          verbose_eval=20)

        rounds = model.best_iteration + 1

        train_rmse = mean_squared_error(dtrain.get_label(), model.predict(dtrain))
        valid_rmse = mean_squared_error(dvalid.get_label(), model.predict(dvalid))
        print 'learning rate = {}, train_rmse = {}, valid_rmse = {}'.format(xgb_params["eta"], train_rmse, valid_rmse)

        if valid_rmse > best_valid_rmse:
            print '---> cv-rmse increased, perform learning rate decay!'
            learning_rate = learning_rate_decay * learning_rate
            xgb_params["eta"] = learning_rate
        else:
            best_valid_rmse = valid_rmse
            best_model = model

            ptrain = best_model.predict(dtrain, ntree_limit=rounds, output_margin=True)
            pvalid = best_model.predict(dvalid, ntree_limit=rounds, output_margin=True)

            dtrain.set_base_margin(ptrain)
            dvalid.set_base_margin(pvalid)

    print 'training done, best_valid_rmse = {}'.format(best_valid_rmse)
    print 'predict submit...'
    y_pred = best_model.predict(dtest)
    y_pred = np.exp(y_pred)
    df_sub = pd.DataFrame({'id': id_test, 'trip_duration': y_pred})
    df_sub.to_csv(Configure.submission_path, index=False)


if __name__ == '__main__':
    print '========== apply xgboost model =========='
    main()
