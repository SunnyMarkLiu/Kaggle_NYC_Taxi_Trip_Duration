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


def main():
    if os.path.exists(Configure.processed_train_path.format('1')):
        return

    train, test = data_utils.load_dataset(op_scope=0)
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    print 'generate geography pca features...'
    generate_pca_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['trip_duration'] = trip_durations
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='1')


if __name__ == '__main__':
    print '========== perform feature engineering =========='
    main()
