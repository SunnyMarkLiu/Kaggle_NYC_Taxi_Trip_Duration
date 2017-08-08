#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-6 下午3:11
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from utils import data_utils


def generate_date_features(train, test):
    # 2016-03-14 17:24:55
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
    test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
    # date
    train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
    test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
    # month
    train['pickup_month'] = train['pickup_datetime'].dt.month
    test['pickup_month'] = train['pickup_datetime'].dt.month
    # day
    train['pickup_day'] = train['pickup_datetime'].dt.day
    test['pickup_day'] = train['pickup_datetime'].dt.day
    # hour
    train['pickup_hour'] = train['pickup_datetime'].dt.hour
    test['pickup_hour'] = train['pickup_datetime'].dt.hour
    # weekofyear
    train['pickup_weekofyear'] = train['pickup_datetime'].dt.weekofyear
    test['pickup_weekofyear'] = train['pickup_datetime'].dt.weekofyear
    # dayofweek
    train['pickup_dayofweek'] = train['pickup_datetime'].dt.dayofweek
    test['pickup_dayofweek'] = train['pickup_datetime'].dt.dayofweek

    train['is_weekend'] = train['pickup_dayofweek'].map(lambda d: (d == 0) | (d == 6))
    test['is_weekend'] = test['pickup_dayofweek'].map(lambda d: (d == 0) | (d == 6))

    train.drop(['pickup_datetime', 'dropoff_datetime', 'pickup_date'], axis=1, inplace=True)
    test.drop(['pickup_datetime', 'pickup_date'], axis=1, inplace=True)


def main():
    train, test = data_utils.load_dataset()
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'generate date features...'
    generate_date_features(train, test)

    # store_and_fwd_flag
    train['is_store_and_fwd_flag'] = train['store_and_fwd_flag'].map(lambda s: s == 'Y')
    test['is_store_and_fwd_flag'] = test['store_and_fwd_flag'].map(lambda s: s == 'Y')
    del train['store_and_fwd_flag']
    del test['store_and_fwd_flag']

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test)


if __name__ == '__main__':
    print '========== perform simple train test preprocess =========='
    main()
