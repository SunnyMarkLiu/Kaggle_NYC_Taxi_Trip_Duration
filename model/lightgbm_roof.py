#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
lightgbm model run-out-of-fold

@author: MarkLiu
@time  : 17-9-12 上午9:40
"""
import gc
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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

    shuffled_index = np.arange(0, train.shape[0], 1)
    np.random.shuffle(shuffled_index)

    # random_indexs = shuffled_index[:int(train.shape[0] * 0.70)]
    random_indexs = shuffled_index
    train = train.iloc[random_indexs, :]

    train['trip_duration'] = np.log(train['trip_duration'])
    y_train_all = train['trip_duration']
    id_train = train['id']
    # del train['id']
    del train['trip_duration']
    id_test = test['id']
    # del test['id']

    train['id'] = train['id'].map(lambda d: int(d[2:]))
    test['id'] = test['id'].map(lambda d: int(d[2:]))

    print 'train:', train.shape, ', test:', test.shape
    print 'feature check before modeling...'
    # feature_util.feature_check_before_modeling(train, test, train.columns)

    X_train = train
    Y_train = y_train_all
    X_test = test

    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'subsample': 0.75,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'min_split_gain': 0.4,

        'num_leaves': 2 ** 6,
        'learning_rate': 0.015,
        'max_depth': 10,

        'reg_alpha': 0.1,
        'reg_lambda': 0.1,

        'scale_pos_weight': 1,
        'early_stopping_round': 20,
        'metric': 'rmsle',
        'verbose': 0
    }

    def lgb_rmsle_score(preds, dtrain):
        labels = np.exp(dtrain.get_label())
        preds = np.exp(preds.clip(min=0))
        return 'rmsle', np.sqrt(np.mean(np.square(np.log1p(preds) - np.log1p(labels)))), False

    roof_flod = 5
    kf = KFold(n_splits=roof_flod, shuffle=True, random_state=42)

    pred_train_full = np.zeros(train.shape[0])
    pred_test_full = 0
    cv_scores = []

    for i, (dev_index, val_index) in enumerate(kf.split(train)):
        print '========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index))
        train_X, val_X = X_train.ix[dev_index], X_train.ix[val_index]
        train_y, val_y = Y_train[dev_index], Y_train[val_index]

        lgb_train = lgbm.Dataset(train_X, label=train_y)
        """ reference=lgb_train is important!"""
        lgb_eval = lgbm.Dataset(val_X, val_y, reference=lgb_train)

        model = lgbm.train(lgbm_params,
                           lgb_train,
                           num_boost_round=20000,
                           valid_sets=lgb_eval,
                           early_stopping_rounds=200,
                           feval=lgb_rmsle_score,
                           verbose_eval=20)

        predict_val = model.predict(val_X)
        predict_test = model.predict(X_test)

        valid_rmse = mean_squared_error(val_y, predict_val)
        print '========== valid_rmse = {} =========='.format(valid_rmse)
        cv_scores.append(valid_rmse)

        # run-out-of-fold predict
        pred_train_full[val_index] = predict_val
        pred_test_full += predict_test

        test_pred_df = pd.DataFrame({'id': id_test})
        test_pred_df['trip_duration'] = np.exp(predict_test)
        test_pred_df.to_csv("lightgbm_roof_fold_{}_predict.csv.gz".format(i), index=False, compression='gzip')

        del lgb_train, lgb_eval, model
        gc.collect()

    print 'Mean cv rmse:', np.mean(cv_scores)
    pred_test_full = pred_test_full / float(roof_flod)

    pred_test_full = np.exp(pred_test_full)
    pred_train_full = np.exp(pred_train_full)

    # saving train predictions for ensemble #
    train_pred_df = pd.DataFrame({'id': id_train})
    train_pred_df['trip_duration'] = pred_train_full
    train_pred_df.to_csv("./model_ensemble/lightgbm_roof_predict_train.csv", index=False)

    # saving test predictions for ensemble #
    test_pred_df = pd.DataFrame({'id': id_test})
    test_pred_df['trip_duration'] = pred_test_full
    test_pred_df.to_csv("./model_ensemble/lightgbm_roof_predict_test.csv", index=False)


if __name__ == '__main__':
    print '========== apply lightgbm model =========='
    main()
