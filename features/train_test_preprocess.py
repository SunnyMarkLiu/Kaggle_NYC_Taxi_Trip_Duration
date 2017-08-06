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


def main():
    train, test = data_utils.load_dataset()
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'generate date features...'
    generate_date_features(train, test)
    print 'train: {}, test: {}'.format(train.shape, test.shape)


if __name__ == '__main__':
    print '========== perform simple train test preprocess =========='
    main()
