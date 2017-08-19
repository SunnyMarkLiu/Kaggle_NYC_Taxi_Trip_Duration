#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-19 下午4:38
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


def calc_heavy_traffic_cluster_distances(conbined_data, n_clusters, batch_size, most_traffic_quantile=0.9,
                                         random_state=42):
    """
    calc the distance between heavy traffic cluster center and pick-up location
    """
    print 'location clustering n_clusters = {}...'.format(n_clusters)
    coords = np.vstack((conbined_data[['pickup_latitude', 'pickup_longitude']].values,
                        conbined_data[['dropoff_latitude', 'dropoff_longitude']].values))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                             random_state=random_state).fit(coords)

    pickup_cluster_fea = 'calc_pickup_kmeans_{}_cluster'.format(n_clusters)
    dropoff_cluster_fea = 'calc_dropoff_kmeans_{}_cluster'.format(n_clusters)

    conbined_data.loc[:, pickup_cluster_fea] = \
        kmeans.predict(conbined_data[['pickup_latitude', 'pickup_longitude']])
    conbined_data.loc[:, dropoff_cluster_fea] = \
        kmeans.predict(conbined_data[['dropoff_latitude', 'dropoff_longitude']])

    # cluster center location
    cluster_centers = kmeans.cluster_centers_

    # calc the traffic groupby cluster
    pickup_cluster_traffic = conbined_data.groupby([pickup_cluster_fea]).count()['id'].reset_index()
    dropoff_cluster_traffic = conbined_data.groupby([dropoff_cluster_fea]).count()['id'].reset_index()
    cluster_traffic = pd.DataFrame({'cluster': pickup_cluster_traffic[pickup_cluster_fea]})
    cluster_traffic['traffic'] = pickup_cluster_traffic['id'] + dropoff_cluster_traffic['id']
    cluster_traffic = cluster_traffic.sort_values(by='traffic', ascending=False)
    most_traffic_clusters = cluster_traffic[cluster_traffic.traffic > \
                                            cluster_traffic.traffic.quantile(most_traffic_quantile)]['cluster']

    print 'calc heavy traffic cluster distances......'
    for most_traffic_cluster in most_traffic_clusters:
        print '>>>calc heavy_traffic_cluster_distances, most_traffic_cluster =', most_traffic_cluster
        conbined_data['pickup_heavy_traffic_cluster{}_haversine'.format(most_traffic_cluster)] = \
            haversine_array(conbined_data['pickup_latitude'].values,
                            conbined_data['pickup_longitude'].values,
                            cluster_centers[most_traffic_cluster][0],
                            cluster_centers[most_traffic_cluster][1])

        conbined_data['pickup_heavy_traffic_cluster{}_bearing'.format(most_traffic_cluster)] = \
            bearing_array(conbined_data['pickup_latitude'].values,
                          conbined_data['pickup_longitude'].values,
                          cluster_centers[most_traffic_cluster][0],
                          cluster_centers[most_traffic_cluster][1])

        conbined_data['dropoff_heavy_traffic_cluster{}_haversine'.format(most_traffic_cluster)] = \
            haversine_array(conbined_data['dropoff_latitude'].values,
                            conbined_data['dropoff_longitude'].values,
                            cluster_centers[most_traffic_cluster][0],
                            cluster_centers[most_traffic_cluster][1])

        conbined_data['dropoff_heavy_traffic_cluster{}_bearing'.format(most_traffic_cluster)] = \
            bearing_array(conbined_data['dropoff_latitude'].values,
                          conbined_data['dropoff_longitude'].values,
                          cluster_centers[most_traffic_cluster][0],
                          cluster_centers[most_traffic_cluster][1])

    conbined_data.drop(['calc_pickup_kmeans_{}_cluster'.format(n_clusters),
                        'calc_dropoff_kmeans_{}_cluster'.format(n_clusters)],
                       axis=1, inplace=True)
    return conbined_data


def main():
    if os.path.exists(Configure.processed_train_path.format('5')):
        return

    train, test = data_utils.load_dataset(op_scope='4')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    n_clusters = 10 ** 2
    conbined_data = calc_heavy_traffic_cluster_distances(conbined_data,
                                                         n_clusters=n_clusters,
                                                         batch_size=64 ** 3,
                                                         most_traffic_quantile=0.9,
                                                         random_state=1000)
    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='5')


if __name__ == '__main__':
    print '========== generate heavy traffic cluster distance =========='
    main()
