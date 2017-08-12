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


def generate_distance_features(train, test, loc1='latitude', loc2='longitude', fea_name='lat_long_'):
    pickup_loc1 = 'pickup_{}'.format(loc1)
    pickup_loc2 = 'pickup_{}'.format(loc2)

    dropoff_loc1 = 'dropoff_{}'.format(loc1)
    dropoff_loc2 = 'dropoff_{}'.format(loc2)

    train.loc[:, fea_name + 'distance_haversine'] = haversine_array(train[pickup_loc1].values,
                                                                    train[pickup_loc2].values,
                                                                    train[dropoff_loc1].values,
                                                                    train[dropoff_loc2].values)
    test.loc[:, fea_name + 'distance_haversine'] = haversine_array(test[pickup_loc1].values,
                                                                   test[pickup_loc2].values,
                                                                   test[dropoff_loc1].values,
                                                                   test[dropoff_loc2].values)
    train.loc[:, fea_name + 'distance_dummy_manhattan'] = dummy_manhattan_distance(train[pickup_loc1].values,
                                                                                   train[pickup_loc2].values,
                                                                                   train[dropoff_loc1].values,
                                                                                   train[dropoff_loc2].values)
    test.loc[:, fea_name + 'distance_dummy_manhattan'] = dummy_manhattan_distance(test[pickup_loc1].values,
                                                                                  test[pickup_loc2].values,
                                                                                  test[dropoff_loc1].values,
                                                                                  test[dropoff_loc2].values)

    train.loc[:, fea_name + 'direction'] = bearing_array(train[pickup_loc1].values, train[pickup_loc2].values,
                                                         train[dropoff_loc1].values, train[dropoff_loc2].values)
    test.loc[:, fea_name + 'direction'] = bearing_array(test[pickup_loc1].values, test[pickup_loc2].values,
                                                        test[dropoff_loc1].values, test[dropoff_loc2].values)

    train.loc[:, fea_name + 'manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + \
                                           np.abs(train['dropoff_pca0'] - train['pickup_pca0'])
    test.loc[:, fea_name + 'manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + \
                                          np.abs(test['dropoff_pca0'] - test['pickup_pca0'])

    train.loc[:, fea_name + 'center_latitude'] = (train[pickup_loc1].values + train[dropoff_loc1].values) / 2
    train.loc[:, fea_name + 'center_longitude'] = (train[pickup_loc2].values + train[dropoff_loc2].values) / 2
    test.loc[:, fea_name + 'center_latitude'] = (test[pickup_loc1].values + test[dropoff_loc1].values) / 2
    test.loc[:, fea_name + 'center_longitude'] = (test[pickup_loc2].values + test[dropoff_loc2].values) / 2


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
    # week_hour
    conbined_data.loc[:, 'pickup_week_hour'] = conbined_data['pickup_weekday'] * 24 + conbined_data['pickup_hour']

    # is_weekend
    conbined_data['is_weekend'] = conbined_data['pickup_weekday'].map(lambda d: 1 if (d == 0) | (d == 6) else 0)

    conbined_data['pickup_time_delta'] = (conbined_data['pickup_datetime'] -
                                          conbined_data['pickup_datetime'].min()).dt.total_seconds()
    conbined_data['pickup_week_delta'] = \
        conbined_data['pickup_weekday'] + \
        ((conbined_data['pickup_hour'] + (conbined_data['pickup_datetime'].dt.minute / 60.0)) / 24.0)

    # Make time features cyclic
    conbined_data['pickup_week_delta_sin'] = np.sin((conbined_data['pickup_week_delta'] / 7) * np.pi)
    conbined_data['pickup_hour_sin'] = np.sin((conbined_data['pickup_hour'] / 24) * np.pi)

    conbined_data.drop(['dropoff_datetime', 'pickup_date'], axis=1, inplace=True)


def generate_location_bin_features(train, test, loc1='latitude', loc2='longitude', fea_name='lat_long_', round_num=2):
    pickup_loc1 = 'pickup_{}'.format(loc1)
    pickup_loc2 = 'pickup_{}'.format(loc2)

    dropoff_loc1 = 'dropoff_{}'.format(loc1)
    dropoff_loc2 = 'dropoff_{}'.format(loc2)

    train.loc[:, '{}_bin'.format(pickup_loc1)] = np.round(train[pickup_loc1], round_num)
    train.loc[:, '{}_bin'.format(pickup_loc2)] = np.round(train[pickup_loc2], round_num)
    test.loc[:, '{}_bin'.format(pickup_loc1)] = np.round(test[pickup_loc1], round_num)
    test.loc[:, '{}_bin'.format(pickup_loc2)] = np.round(test[pickup_loc2], round_num)

    train.loc[:, '{}_bin'.format(dropoff_loc1)] = np.round(train[dropoff_loc1], round_num)
    train.loc[:, '{}_bin'.format(dropoff_loc2)] = np.round(train[dropoff_loc2], round_num)
    test.loc[:, '{}_bin'.format(dropoff_loc1)] = np.round(test[dropoff_loc1], round_num)
    test.loc[:, '{}_bin'.format(dropoff_loc2)] = np.round(test[dropoff_loc2], round_num)

    train.loc[:, '{}center_latitude_bin'.format(fea_name)] = np.round(train['{}center_latitude'.format(fea_name)],
                                                                      round_num)
    train.loc[:, '{}center_longitude_bin'.format(fea_name)] = np.round(train['{}center_longitude'.format(fea_name)],
                                                                       round_num)
    test.loc[:, '{}center_latitude_bin'.format(fea_name)] = np.round(test['{}center_latitude'.format(fea_name)],
                                                                     round_num)
    test.loc[:, '{}center_longitude_bin'.format(fea_name)] = np.round(test['{}center_longitude'.format(fea_name)],
                                                                      round_num)


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
    generate_distance_features(train, test, loc1='latitude', loc2='longitude', fea_name='lat_long_')

    print 'generate pca distance features...'
    generate_distance_features(train, test, loc1='pca0', loc2='pca1', fea_name='pca_')

    print 'generate location bin features...'
    generate_location_bin_features(train, test, loc1='latitude', loc2='longitude',
                                   fea_name='lat_long_', round_num=2)
    print 'generate pca location bin features...'
    # train['pickup_pca0'] = train['pickup_pca0'] * 1000
    # train['pickup_pca1'] = train['pickup_pca1'] * 1000
    # test['pickup_pca0'] = test['pickup_pca0'] * 1000
    # test['pickup_pca1'] = test['pickup_pca1'] * 1000
    # generate_location_bin_features(train, test, loc1='pca0', loc2='pca1',
    #                                fea_name='pca_', round_num=2)
    # train['pickup_pca0'] = train['pickup_pca0'] / 1000
    # train['pickup_pca1'] = train['pickup_pca1'] / 1000
    # test['pickup_pca0'] = test['pickup_pca0'] / 1000
    # test['pickup_pca1'] = test['pickup_pca1'] / 1000

    train['trip_duration'] = trip_durations
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='1')


if __name__ == '__main__':
    print '========== perform feature engineering =========='
    main()
