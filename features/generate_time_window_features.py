#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-12 下午4:01
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

import datetime

# my own module
from utils import data_utils
from conf.configure import Configure


def generate_pikcup_timewindow_traffic(conbined_data_df, timewindow_days, n_clusters):
    """
    获取出发地所在 cluster 之后 timewindow 时间内的 traffic
    """
    cluster_feature = 'pickup_kmeans_{}_cluster'.format(n_clusters)
    conbined_data = conbined_data_df.copy()[['id', 'pickup_datetime', cluster_feature]]

    timewindow_traffic = pd.DataFrame()
    timewindow_traffic['id'] = conbined_data['id']
    for timewindow in timewindow_days:
        print 'perform timewindow =', timewindow
        cache_file = Configure.pikcup_time_window_cluster_traffic_features_path.format(timewindow, n_clusters)

        if not os.path.exists(cache_file):
            after_timewindow_traffic = []
            for i in tqdm(range(conbined_data.shape[0])):
                current_time = conbined_data.loc[i, 'pickup_datetime']
                indexs = (current_time + datetime.timedelta(minutes=timewindow) < conbined_data['pickup_datetime']) & \
                         (conbined_data['pickup_datetime'] < current_time)
                df = conbined_data[indexs]
                if df.shape[0] == 0:
                    after_timewindow_traffic.append(0)
                    continue
                df = df.groupby([cluster_feature]).count()['pickup_datetime'].reset_index()
                df.columns = [cluster_feature, 'traffic_count']

                traffic_count = df[df[cluster_feature] == conbined_data.loc[i, cluster_feature]]['traffic_count'].values
                traffic_count = 0 if len(traffic_count) == 0 else traffic_count[0]
                after_timewindow_traffic.append(traffic_count)

            feature = 'this_' + cluster_feature + 'after_' + str(timewindow) + 'minutes_traffic'
            after_features_df = pd.DataFrame({'id': conbined_data['id'],
                                              feature: after_timewindow_traffic})

            after_features_df.to_csv(cache_file, index=False)
        else:
            after_features_df = pd.read_csv(cache_file)

        timewindow_traffic = pd.merge(timewindow_traffic, after_features_df, how='left', on='id')

    return timewindow_traffic


def generate_dropoff_timewindow_traffic(conbined_data_df, timewindow_days, n_clusters):
    """
    获取目的地所在 cluster 之前 timewindow 时间内的 traffic
    """
    cluster_feature = 'dropoff_kmeans_{}_cluster'.format(n_clusters)
    conbined_data = conbined_data_df.copy()[['id', 'dropoff_datetime', cluster_feature]]

    timewindow_traffic = pd.DataFrame()
    timewindow_traffic['id'] = conbined_data['id']
    for timewindow in timewindow_days:
        print 'perform timewindow =', timewindow
        cache_file = Configure.dropoff_time_window_cluster_traffic_features_path.format(timewindow, n_clusters)

        if not os.path.exists(cache_file):
            pre_timewindow_traffic = []
            for i in tqdm(range(conbined_data.shape[0])):
                current_time = conbined_data.loc[i, 'dropoff_datetime']
                indexs = (current_time < conbined_data['dropoff_datetime']) & \
                         (conbined_data['dropoff_datetime'] < (current_time + datetime.timedelta(minutes=timewindow)))
                df = conbined_data[indexs]
                if df.shape[0] == 0:
                    pre_timewindow_traffic.append(0)
                    continue
                df = df.groupby([cluster_feature]).count()['dropoff_datetime'].reset_index()
                df.columns = [cluster_feature, 'traffic_count']

                traffic_count = df[df[cluster_feature] == conbined_data.loc[i, cluster_feature]]['traffic_count'].values
                traffic_count = 0 if len(traffic_count) == 0 else traffic_count[0]
                pre_timewindow_traffic.append(traffic_count)

            feature = 'this_' + cluster_feature + 'pre_' + str(timewindow) + 'minutes_traffic'
            pre_features_df = pd.DataFrame({'id': conbined_data['id'],
                                            feature: pre_timewindow_traffic})

            pre_features_df.to_csv(cache_file, index=False)
        else:
            pre_features_df = pd.read_csv(cache_file)

        timewindow_traffic = pd.merge(timewindow_traffic, pre_features_df, how='left', on='id')

    return timewindow_traffic


def perform_time_window(conbined_data, timewindow_days):
    """应用时间窗"""
    n_clusters = 100
    print 'generate pikcup timewindow traffic...'
    pikcup_timewindow_traffic = generate_pikcup_timewindow_traffic(conbined_data, timewindow_days, n_clusters)
    print 'generate dropoff timewindow traffic...'
    dropoff_timewindow_traffic = generate_dropoff_timewindow_traffic(conbined_data, timewindow_days, n_clusters)
    conbined_data = pd.merge(conbined_data, pikcup_timewindow_traffic, how='left', on='id')
    conbined_data = pd.merge(conbined_data, dropoff_timewindow_traffic, how='left', on='id')

    return conbined_data


def main():
    if os.path.exists(Configure.processed_train_path.format('5')):
        return
    train, test = data_utils.load_dataset(op_scope='4')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])
    conbined_data.columns = test.columns.values
    conbined_data.index = range(conbined_data.shape[0])

    # timewindow size in minutes
    timewindow_days = [3, 5, 10, 15, 30]
    conbined_data = perform_time_window(conbined_data, timewindow_days)

    conbined_data.drop(['pickup_datetime', 'dropoff_datetime'], axis=1, inplace=True)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['trip_duration'] = trip_durations
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='5')


if __name__ == '__main__':
    print "========== apply time window generate some statistic features =========="
    main()
