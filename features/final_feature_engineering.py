#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-10 下午7:18
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


def main():
    if os.path.exists(Configure.processed_train_path.format('7')):
        return

    train, test = data_utils.load_dataset(op_scope='6')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    conbined_data['is_holyday'] = conbined_data.apply(
        lambda row: 1 if (row['pickup_month'] == 1 and row['pickup_day'] == 1) or (
        row['pickup_month'] == 7 and row['pickup_day'] == 4) or (
                             row['pickup_month'] == 11 and row['pickup_day'] == 11) or (
                         row['pickup_month'] == 12 and row['pickup_day'] == 25) or (
                             row['pickup_month'] == 1 and row['pickup_day'] >= 15 and row['pickup_day'] <= 21 and row[
                                 'pickup_weekday'] == 0) or (
                             row['pickup_month'] == 2 and row['pickup_day'] >= 15 and row['pickup_day'] <= 21 and row[
                                 'pickup_weekday'] == 0) or (
                             row['pickup_month'] == 5 and row['pickup_day'] >= 25 and row['pickup_day'] <= 31 and row[
                                 'pickup_weekday'] == 0) or (
                             row['pickup_month'] == 9 and row['pickup_day'] >= 1 and row['pickup_day'] <= 7 and row[
                                 'pickup_weekday'] == 0) or (
                             row['pickup_month'] == 10 and row['pickup_day'] >= 8 and row['pickup_day'] <= 14 and row[
                                 'pickup_weekday'] == 0) or (
                             row['pickup_month'] == 11 and row['pickup_day'] >= 22 and row['pickup_day'] <= 28 and row[
                                 'pickup_weekday'] == 3) else 0,
        axis=1)
    conbined_data['is_day_before_holyday'] = conbined_data.apply(
        lambda row: 1 if (row['pickup_month'] == 12 and row['pickup_day'] == 31) or (
        row['pickup_month'] == 7 and row['pickup_day'] == 3) or (
                             row['pickup_month'] == 11 and row['pickup_day'] == 10) or (
                         row['pickup_month'] == 12 and row['pickup_day'] == 24) or (
                             row['pickup_month'] == 1 and row['pickup_day'] >= 14 and row['pickup_day'] <= 20 and row[
                                 'pickup_weekday'] == 6) or (
                             row['pickup_month'] == 2 and row['pickup_day'] >= 14 and row['pickup_day'] <= 20 and row[
                                 'pickup_weekday'] == 6) or (
                             row['pickup_month'] == 5 and row['pickup_day'] >= 24 and row['pickup_day'] <= 30 and row[
                                 'pickup_weekday'] == 6) or (
                             (row['pickup_month'] == 9 and row['pickup_day'] >= 1 and row['pickup_day'] <= 6) or (
                                 row['pickup_month'] == 8 and row['pickup_day'] == 31) and row[
                                 'pickup_weekday'] == 6) or (
                             row['pickup_month'] == 10 and row['pickup_day'] >= 7 and row['pickup_day'] <= 13 and row[
                                 'pickup_weekday'] == 6) or (
                             row['pickup_month'] == 11 and row['pickup_day'] >= 21 and row['pickup_day'] <= 27 and row[
                                 'pickup_weekday'] == 2) else 0,
        axis=1)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='7')


if __name__ == '__main__':
    print '========== final feature engineering =========='
    main()
