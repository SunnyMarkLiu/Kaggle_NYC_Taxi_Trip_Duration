#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-5 下午8:41
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from utils import data_utils

# remove warnings
import warnings

warnings.filterwarnings('ignore')


def generate_binary_features(conbined_data):
    """generate some binary features"""
    conbined_data['is_passenger_alone'] = conbined_data['passenger_count'].map(lambda pc: 1 if pc == 1 else 0)
    conbined_data['is_midnight'] = conbined_data['pickup_hour'].map(lambda ph: 1 if (ph >= 0) and (ph <= 5) else 0)


def main():
    train, test = data_utils.load_dataset(op_scope='4')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    generate_binary_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['trip_duration'] = trip_durations

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='5')


if __name__ == '__main__':
    print '========== perform other feature engineering =========='
    main()
