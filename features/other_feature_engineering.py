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

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from utils import data_utils

# remove warnings
import warnings

warnings.filterwarnings('ignore')


def generate_binary_features(conbined_data):
    """generate some binary features"""
    conbined_data['is_passenger_alone'] = conbined_data['passenger_count'].map(lambda pc: 1 if pc == 1 else 0)
    conbined_data['is_midnight'] = conbined_data['pickup_hour'].map(lambda ph: 1 if (ph >= 0) and (ph <= 5) else 0)

def location_clustering(conbined_data, n_clusters, batch_size, random_state=42):
    coords = np.vstack((conbined_data[['pickup_latitude', 'pickup_longitude']].values,
                        conbined_data[['dropoff_latitude', 'dropoff_longitude']].values))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                             random_state=random_state).fit(coords)
    conbined_data.loc[:, 'pickup_kmeans_{}_cluster'.format(n_clusters)] = \
        kmeans.predict(conbined_data[['pickup_latitude', 'pickup_longitude']])
    conbined_data.loc[:, 'dropoff_kmeans_{}_cluster'.format(n_clusters)] = \
        kmeans.predict(conbined_data[['dropoff_latitude', 'dropoff_longitude']])

def generate_groupby_speed_features(train, test, n_clusters, loc1='latitude', loc2='longitude', fea_name='lat_long_'):
    train.loc[:, fea_name + 'avg_speed_h'] = 1000 * train[fea_name + 'distance_haversine'] / train['trip_duration']
    train.loc[:, fea_name + 'avg_speed_m'] = 1000 * train[fea_name + 'distance_dummy_manhattan'] / train['trip_duration']

    train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
    target_columns = [fea_name + 'avg_speed_h', fea_name + 'avg_speed_m', 'log_trip_duration']
    # groupby
    gby_cols = ['pickup_weekofyear', 'pickup_hour', 'pickup_weekday', 'pickup_week_hour',
                'pickup_week_delta_sin', 'pickup_hour_sin', 'pickup_day',
                'pickup_kmeans_{}_cluster'.format(n_clusters),
                'dropoff_kmeans_{}_cluster'.format(n_clusters)]

    free_memory = ['id'] + gby_cols + target_columns
    for gby_col in gby_cols:
        print '>>>> perform groupby {}...'.format(gby_col)
        gby = train[free_memory].groupby(gby_col).mean()[target_columns]
        gby.columns = ['%s_gby_%s_n_clusters_%s' % (col, gby_col, n_clusters) for col in gby.columns]
        train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
        test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)
    print 'perform groupby done.'
    pickup_loc1 = 'pickup_{}'.format(loc1)
    pickup_loc2 = 'pickup_{}'.format(loc2)

    gby_colses = [[pickup_loc1+'_bin', pickup_loc2+'_bin'],
                ['pickup_week_hour', pickup_loc1+'_bin', pickup_loc2+'_bin'],
                ['pickup_hour', pickup_loc1+'_bin', pickup_loc2+'_bin'],
                ['pickup_kmeans_{}_cluster'.format(n_clusters), 'dropoff_kmeans_{}_cluster'.format(n_clusters)],
                ['pickup_week_hour', 'pickup_kmeans_{}_cluster'.format(n_clusters), 'dropoff_kmeans_{}_cluster'.format(n_clusters)],
                ['pickup_hour', 'pickup_kmeans_{}_cluster'.format(n_clusters), 'dropoff_kmeans_{}_cluster'.format(n_clusters)],
                ['pickup_hour', 'pickup_kmeans_{}_cluster'.format(n_clusters)],
                ['pickup_hour', 'dropoff_kmeans_{}_cluster'.format(n_clusters)]]

    for gby_cols in gby_colses:
        print '>>>> perform multi-groupby {}...'.format(gby_cols)
        freed_columns = target_columns
        freed_columns.extend(gby_cols)
        freed_columns.append('id')
        freed_columns = list(set(freed_columns))
        coord_speed = train[freed_columns].groupby(gby_cols).mean()[[fea_name + 'avg_speed_h']].reset_index()
        coord_count = train[freed_columns].groupby(gby_cols).count()[['id']].reset_index()
        coord_speed2 = train[freed_columns].groupby(gby_cols).mean()[[fea_name + 'avg_speed_m']].reset_index()
        coord_stats = pd.merge(coord_speed, coord_count, how='left', on=gby_cols)
        coord_stats = pd.merge(coord_stats, coord_speed2, how='left', on=gby_cols)

        coord_stats = coord_stats[coord_stats['id'] > 100]
        coord_stats.columns = gby_cols + [fea_name + 'avg_speed_h_%s_n_clusters_%s' % ('_'.join(gby_cols), n_clusters), 'cnt_%s_n_clusters_%s' % ('_'.join(gby_cols), n_clusters)] + \
                              [fea_name + 'avg_speed_m_%s_n_clusters_%s' % ('_'.join(gby_cols), n_clusters)]

        train = pd.merge(train, coord_stats, how='left', on=gby_cols)
        test = pd.merge(test, coord_stats, how='left', on=gby_cols)

    print 'perform multi-groupby done.'
    drop_columns = [fea_name + 'avg_speed_h', fea_name + 'avg_speed_m', 'log_trip_duration']
    train.drop(drop_columns, axis=1, inplace=True)

    return train, test

def main():
    train, test = data_utils.load_dataset(op_scope='4')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    generate_binary_features(conbined_data)

    for n_clusters in [6**2]:
        print 'location clustering n_clusters = {}...'.format(n_clusters)
        location_clustering(conbined_data, n_clusters=n_clusters, batch_size=64 ** 3, random_state=1000)

        train = conbined_data.iloc[:train.shape[0], :]
        test = conbined_data.iloc[train.shape[0]:, :]
        train['trip_duration'] = trip_durations

        print 'generate lat_long groupby speed features...'
        train, test = generate_groupby_speed_features(train, test, n_clusters, loc1='latitude', loc2='longitude',
                                                      fea_name='lat_long_')
        del train['trip_duration']
        print 'train: {}, test: {}'.format(train.shape, test.shape)
        conbined_data = pd.concat([train, test])

    train['trip_duration'] = trip_durations
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='5')


if __name__ == '__main__':
    print '========== perform other feature engineering =========='
    main()
