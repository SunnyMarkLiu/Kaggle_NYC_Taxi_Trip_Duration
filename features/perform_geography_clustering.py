#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-10 下午4:11
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from utils import data_utils
from conf.configure import Configure
# remove warnings
import warnings

warnings.filterwarnings('ignore')


def location_clustering(conbined_data, n_clusters, batch_size):
    coords = np.vstack((conbined_data[['pickup_latitude', 'pickup_longitude']].values,
                        conbined_data[['dropoff_latitude', 'dropoff_longitude']].values))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size).fit(coords)
    conbined_data.loc[:, 'pickup_kmeans_{}_cluster'.format(n_clusters)] = \
        kmeans.predict(conbined_data[['pickup_latitude', 'pickup_longitude']])
    conbined_data.loc[:, 'dropoff_kmeans_{}_cluster'.format(n_clusters)] = \
        kmeans.predict(conbined_data[['dropoff_latitude', 'dropoff_longitude']])


def generate_groupby_speed_features(train, test, n_clusters, fea_name='lat_long_'):
    train.loc[:, fea_name + 'avg_speed_h'] = 1000 * train[fea_name + 'distance_haversine'] / train['trip_duration']
    train.loc[:, fea_name + 'avg_speed_m'] = 1000 * train[fea_name + 'distance_dummy_manhattan'] / train[
        'trip_duration']

    train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
    # groupby
    gby_cols = ['pickup_weekofyear', 'pickup_hour', 'pickup_weekday', 'pickup_week_hour',
                'pickup_kmeans_{}_cluster'.format(n_clusters), 'dropoff_kmeans_{}_cluster'.format(n_clusters)]
    for gby_col in gby_cols:
        gby = train.groupby(gby_col).mean()[[fea_name + 'avg_speed_h', fea_name + 'avg_speed_m', 'log_trip_duration']]
        gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
        train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
        test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

    drop_columns = [fea_name + 'avg_speed_h', fea_name + 'avg_speed_m', 'log_trip_duration']
    train.drop(drop_columns, axis=1, inplace=True)

    return train, test


def main():
    if os.path.exists(Configure.processed_train_path.format('3')):
        return

    train, test = data_utils.load_dataset(op_scope='2')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    n_clusters = 10 ** 2
    print 'location clustering...'
    location_clustering(conbined_data, n_clusters=n_clusters, batch_size=32 ** 3)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    print 'generate lat_long groupby speed features...'
    train, test = generate_groupby_speed_features(train, test, n_clusters, fea_name='lat_long_')
    print 'generate pca groupby speed features...'
    train, test = generate_groupby_speed_features(train, test, n_clusters, fea_name='pca_')

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='3')


if __name__ == '__main__':
    print '========== perform geography clustering =========='
    main()
