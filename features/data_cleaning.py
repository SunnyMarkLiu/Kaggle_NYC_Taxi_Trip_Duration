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
import pandas as pd
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
    train = train[train['lat_long_distance_haversine'] < 200]
    # train = train[train['trip_duration'] < 500000] # 导致过拟合

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='2')


if __name__ == '__main__':
    print '========== perform data cleaning =========='
    main()
