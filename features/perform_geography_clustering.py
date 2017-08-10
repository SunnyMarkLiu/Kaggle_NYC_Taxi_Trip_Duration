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


def main():
    if os.path.exists(Configure.processed_train_path.format('3')):
        return

    train, test = data_utils.load_dataset(op_scope='2')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    location_clustering(conbined_data, n_clusters=10 ** 2, batch_size=32 ** 3)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='3')


if __name__ == '__main__':
    print '========== perform geography clustering =========='
    main()
