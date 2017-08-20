#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-17 下午4:00
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd

from utils import data_utils
from conf.configure import Configure
# remove warnings
import warnings

warnings.filterwarnings('ignore')


def drop_some_features(conbined_data, drop_missing_rate=0.9):
    missing_df = conbined_data.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_rate'] = 1.0 * missing_df['missing_count'] / conbined_data.shape[0]
    missing_df = missing_df[missing_df.missing_count > 0]
    missing_df = missing_df.sort_values(by='missing_count', ascending=False)

    drop_columns = missing_df[missing_df['missing_rate'] > drop_missing_rate]['column_name'].values
    conbined_data.drop(drop_columns, axis=1, inplace=True)
    return conbined_data


def main():
    train, test = data_utils.load_dataset(op_scope='3')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    drop_missing_rate = 1
    print 'drop some features, missing_rate > {}'.format(drop_missing_rate)
    conbined_data = drop_some_features(conbined_data, drop_missing_rate=drop_missing_rate)

    conbined_data.drop(['pickup_datetime', 'dropoff_datetime'], axis=1, inplace=True)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='4')


if __name__ == '__main__':
    print '========== drop some features =========='
    main()
