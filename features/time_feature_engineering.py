#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-12 上午10:42
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


def generate_time_traffic(train, test, n_clusters):
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

    df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_kmeans_{}_cluster'.format(n_clusters),
                                       'dropoff_kmeans_{}_cluster'.format(n_clusters)]]
    df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
    group_freq = '60min'
    train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
    test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

    df_counts['count_60min'] = df_counts.isnull().rolling('60min').count()['id']
    train = train.merge(df_counts, on='id', how='left')
    test = test.merge(df_counts, on='id', how='left')

    dropoff_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.TimeGrouper(group_freq), 'dropoff_kmeans_{}_cluster'.format(n_clusters)]) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('dropoff_kmeans_{}_cluster'.format(n_clusters)).rolling('240min').mean() \
        .drop('dropoff_kmeans_{}_cluster'.format(n_clusters), axis=1) \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(
        columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_{}_cluster_count'.format(n_clusters)})

    train['dropoff_{}_cluster_count'.format(n_clusters)] = \
    train[['pickup_datetime_group', 'dropoff_kmeans_{}_cluster'.format(n_clusters)]].merge(dropoff_counts, on=[
        'pickup_datetime_group', 'dropoff_kmeans_{}_cluster'.format(n_clusters)], how='left')[
        'dropoff_{}_cluster_count'.format(n_clusters)].fillna(0)
    test['dropoff_{}_cluster_count'.format(n_clusters)] = \
    test[['pickup_datetime_group', 'dropoff_kmeans_{}_cluster'.format(n_clusters)]].merge(dropoff_counts, on=[
        'pickup_datetime_group', 'dropoff_kmeans_{}_cluster'.format(n_clusters)], how='left')[
        'dropoff_{}_cluster_count'.format(n_clusters)].fillna(0)

    # Count how many trips are going from each cluster over time
    df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_kmeans_{}_cluster'.format(n_clusters), 'dropoff_kmeans_{}_cluster'.format(n_clusters)]]
    pickup_counts = df_all \
        .set_index('pickup_datetime') \
        .groupby([pd.TimeGrouper(group_freq), 'pickup_kmeans_{}_cluster'.format(n_clusters)]) \
        .agg({'id': 'count'}) \
        .reset_index().set_index('pickup_datetime') \
        .groupby('pickup_kmeans_{}_cluster'.format(n_clusters)).rolling('240min').mean() \
        .drop('pickup_kmeans_{}_cluster'.format(n_clusters), axis=1) \
        .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
        .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_{}_cluster_count'.format(n_clusters)})

    train['pickup_{}_cluster_count'.format(n_clusters)] = train[['pickup_datetime_group', 'pickup_kmeans_{}_cluster'.format(n_clusters)]].merge(pickup_counts, on=[
        'pickup_datetime_group', 'pickup_kmeans_{}_cluster'.format(n_clusters)], how='left')['pickup_{}_cluster_count'.format(n_clusters)].fillna(0)
    test['pickup_{}_cluster_count'.format(n_clusters)] = test[['pickup_datetime_group', 'pickup_kmeans_{}_cluster'.format(n_clusters)]].merge(pickup_counts,
                                                                                           on=['pickup_datetime_group',
                                                                                               'pickup_kmeans_{}_cluster'.format(n_clusters)],
                                                                                           how='left')[
        'pickup_{}_cluster_count'.format(n_clusters)].fillna(0)
    
    train.drop(['pickup_datetime', 'pickup_datetime_group'], axis=1, inplace=True)
    test.drop(['pickup_datetime', 'pickup_datetime_group'], axis=1, inplace=True)

    return train, test


def main():
    if os.path.exists(Configure.processed_train_path.format('4')):
        return

    train, test = data_utils.load_dataset(op_scope='3')
    print 'train: {}, test: {}'.format(train.shape, test.shape)

    # train, test = generate_time_traffic(train, test, n_clusters=100)
    train.drop(['pickup_datetime'], axis=1, inplace=True)
    test.drop(['pickup_datetime'], axis=1, inplace=True)

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='4')


if __name__ == '__main__':
    print '========== perform time feature engineering =========='
    main()
