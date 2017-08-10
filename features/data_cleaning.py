#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-10 上午11:21
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
from utils import data_utils

# remove warnings
import warnings

warnings.filterwarnings('ignore')
from conf.configure import Configure


def main():
    if os.path.exists(Configure.processed_train_path.format('2')):
        return

    train, test = data_utils.load_dataset(op_scope='1')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'data clean according to lat_long_distance_haversine & trip_duration...'
    # train = train[train['lat_long_distance_haversine'] < 300]
    # train = train[train['trip_duration'] <= 1800000].reset_index(drop=True) # 导致过拟合

    print 'train: {}, test: {}'.format(train.shape, test.shape)

    # optimize dtypes
    print('Memory usage, Mb: {:.2f}'.format(train.memory_usage().sum() / 2 ** 20))
    print 'optimize dtypes...'
    train['is_store_and_fwd_flag'] = train['is_store_and_fwd_flag'].astype(np.uint8)
    train['passenger_count'] = train['passenger_count'].astype(np.uint8)
    train['vendor_id'] = train['vendor_id'].astype(np.uint8)
    train['pickup_month'] = train['pickup_month'].astype(np.uint8)
    train['pickup_day'] = train['pickup_day'].astype(np.uint8)
    train['pickup_hour'] = train['pickup_hour'].astype(np.uint8)
    train['pickup_weekofyear'] = train['pickup_weekofyear'].astype(np.uint8)
    train['pickup_weekday'] = train['pickup_weekday'].astype(np.uint8)
    train['is_weekend'] = train['is_weekend'].astype(np.uint8)
    train['trip_duration'] = train['trip_duration'].astype(np.uint32)
    print('After optimized memory usage, Mb: {:.2f}'.format(train.memory_usage().sum() / 2 ** 20))

    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='2')


if __name__ == '__main__':
    print '========== perform data cleaning =========='
    main()
