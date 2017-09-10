#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
xgboost model run-out-of-fold
@author: MarkLiu
@time  : 17-9-10 上午10:44
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import callback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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

    train['trip_duration'] = np.log(train['trip_duration'])
    y_train_all = train['trip_duration']
    id_train = train['id']
    del train['id']
    del train['trip_duration']
    id_test = test['id']
    del test['id']

    print 'train:', train.shape, ', test:', test.shape

    print 'feature check before modeling...'
    # feature_util.feature_check_before_modeling(train, test, train.columns)

    X_train = train
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
        'gpu_id': 0,
        'nthread': -1,
        'silent': 1
    }

    roof_flod = 5
    kf = KFold(n_splits=roof_flod, shuffle=True, random_state=42)
    learning_rates = [0.005] * 2000 + [0.002] * 1000 + [0.001] * 1000 + [0.0005] * 2000

    pred_train_full = np.zeros(train.shape[0])
    pred_test_full = 0
    cv_scores = []

    for i, (dev_index, val_index) in enumerate(kf.split(X_train)):
        print '========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index))
        dev_X, val_X = X_train.ix[dev_index], X_train.ix[val_index]
        dev_y, val_y = y_train_all[dev_index], y_train_all[val_index]
        ddev = xgb.DMatrix(dev_X, dev_y, feature_names=df_columns)
        dval = xgb.DMatrix(val_X, val_y, feature_names=df_columns)

        model = xgb.train(dict(xgb_params), ddev,
                          num_boost_round=6000,
                          evals=[(ddev, 'train'), (dval, 'valid')],
                          early_stopping_rounds=60,
                          verbose_eval=20,
                          callbacks=[callback.reset_learning_rate(learning_rates)])

        pred_valid = model.predict(dval, num_iteration=model.best_iteration)
        pred_test = model.predict(dtest, num_iteration=model.best_iteration)

        valid_rmse = mean_squared_error(dval.get_label(), pred_valid)
        print '========== valid_rmse = {} =========='.format(valid_rmse)
        cv_scores.append(valid_rmse)

        # run-out-of-fold predict
        pred_train_full[val_index] = pred_valid
        pred_test_full += pred_test

    print 'Mean cv rmse:', np.mean(cv_scores)
    pred_test_full = pred_test_full / float(roof_flod)

    pred_test_full = np.exp(pred_test_full)
    pred_train_full = np.exp(pred_train_full)

    # saving train predictions for ensemble #
    train_pred_df = pd.DataFrame({'id': id_train})
    train_pred_df['trip_duration'] = pred_train_full
    train_pred_df.to_csv("train_preds_xgboost.csv", index=False)

    # saving test predictions for ensemble #
    test_pred_df = pd.DataFrame({'id': id_test})
    test_pred_df['trip_duration'] = pred_test_full
    test_pred_df.to_csv("test_preds_xgboost.csv", index=False)

if __name__ == '__main__':
    print '========== apply xgboost model =========='
    main()
