#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-12 上午11:10
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from geopy.distance import great_circle
from utils import data_utils
from conf.configure import Configure
# remove warnings
import warnings

warnings.filterwarnings('ignore')


def main():
    if os.path.exists(Configure.processed_train_path.format('8')):
        return

    train, test = data_utils.load_dataset(op_scope='7')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    def driving_distance(raw):
        startpoint = (raw['pickup_latitude'], raw['pickup_longitude'])
        endpoint = (raw['dropoff_latitude'], raw['dropoff_longitude'])
        distance = great_circle(startpoint, endpoint).miles
        return distance

    print 'calc geopy distance features...'
    conbined_data['osmnx_distance'] = conbined_data[['pickup_latitude', 'pickup_longitude',
                                                     'dropoff_latitude', 'dropoff_longitude']].apply(driving_distance,
                                                                                                     axis=1)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='8')


if __name__ == '__main__':
    print '========== generate geopy distance features =========='
    main()
