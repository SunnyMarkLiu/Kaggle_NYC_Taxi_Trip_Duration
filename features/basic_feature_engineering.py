#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-8 下午1:24
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from utils import data_utils

# remove warnings
import warnings

warnings.filterwarnings('ignore')
from conf.configure import Configure


def generate_pca_features(conbined_data):
    coords = np.vstack((conbined_data[['pickup_latitude', 'pickup_longitude']].values,
                        conbined_data[['dropoff_latitude', 'dropoff_longitude']].values))

    pca = PCA(n_components=2).fit(coords)
    pickip_tr = pca.transform(conbined_data[['pickup_latitude', 'pickup_longitude']])
    conbined_data['pickup_pca0'] = pickip_tr[:, 0]
    conbined_data['pickup_pca1'] = pickip_tr[:, 1]
    dropoff_tr = pca.transform(conbined_data[['dropoff_latitude', 'dropoff_longitude']])
    conbined_data['dropoff_pca0'] = dropoff_tr[:, 0]
    conbined_data['dropoff_pca1'] = dropoff_tr[:, 1]


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def generate_distance_features(train, test):
    train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values,
                                                         train['pickup_longitude'].values,
                                                         train['dropoff_latitude'].values,
                                                         train['dropoff_longitude'].values)
    train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                        train['pickup_longitude'].values,
                                                                        train['dropoff_latitude'].values,
                                                                        train['dropoff_longitude'].values)
    train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                              train['dropoff_latitude'].values, train['dropoff_longitude'].values)
    train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + \
                                    np.abs(train['dropoff_pca0'] - train['pickup_pca0'])

    test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                                        test['dropoff_latitude'].values,
                                                        test['dropoff_longitude'].values)
    test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values,
                                                                       test['pickup_longitude'].values,
                                                                       test['dropoff_latitude'].values,
                                                                       test['dropoff_longitude'].values)
    test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                             test['dropoff_latitude'].values, test['dropoff_longitude'].values)
    test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(
        test['dropoff_pca0'] - test['pickup_pca0'])

    train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
    train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
    test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
    test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2


def generate_date_features(conbined_data):
    # 2016-03-14 17:24:55
    conbined_data['pickup_datetime'] = pd.to_datetime(conbined_data['pickup_datetime'])
    conbined_data['dropoff_datetime'] = pd.to_datetime(conbined_data['dropoff_datetime'])
    # date
    conbined_data.loc[:, 'pickup_date'] = conbined_data['pickup_datetime'].dt.date
    # month
    conbined_data['pickup_month'] = conbined_data['pickup_datetime'].dt.month
    # day
    conbined_data['pickup_day'] = conbined_data['pickup_datetime'].dt.day
    # hour
    conbined_data['pickup_hour'] = conbined_data['pickup_datetime'].dt.hour
    # weekofyear
    conbined_data['pickup_weekofyear'] = conbined_data['pickup_datetime'].dt.weekofyear
    # weekday
    conbined_data['pickup_weekday'] = conbined_data['pickup_datetime'].dt.weekday
    # is_weekend
    conbined_data['is_weekend'] = conbined_data['pickup_weekday'].map(lambda d: (d == 0) | (d == 6))

    conbined_data['pickup_time_delta'] = (conbined_data['pickup_datetime'] -
                                          conbined_data['pickup_datetime'].min()).dt.total_seconds()
    conbined_data['pickup_week_delta'] = \
        conbined_data['pickup_weekday'] + \
        ((conbined_data['pickup_hour'] + (conbined_data['pickup_datetime'].dt.minute / 60.0)) / 24.0)

    # Make time features cyclic
    conbined_data['pickup_week_delta_sin'] = np.sin((conbined_data['pickup_week_delta'] / 7) * np.pi)
    conbined_data['pickup_hour_sin'] = np.sin((conbined_data['pickup_hour'] / 24) * np.pi)

    conbined_data.drop(['pickup_datetime', 'dropoff_datetime', 'pickup_date'], axis=1, inplace=True)


def main():
    if os.path.exists(Configure.processed_train_path.format('1')):
        return

    train, test = data_utils.load_dataset(op_scope='0')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    print 'generate geography pca features...'
    generate_pca_features(conbined_data)

    print 'generate datetime features...'
    generate_date_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    print 'generate distance features...'
    generate_distance_features(train, test)

    train['trip_duration'] = trip_durations
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='1')


if __name__ == '__main__':
    print '========== perform feature engineering =========='
    main()
