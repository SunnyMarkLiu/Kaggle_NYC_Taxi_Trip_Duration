#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-6 下午9:12
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from utils import data_utils

# remove warnings
import warnings

warnings.filterwarnings('ignore')


def generate_street_heavy(train, test):
    starting_street_heavy = train.groupby(['starting_street'])['trip_duration'].mean().reset_index()
    street_heavy_map = starting_street_heavy.set_index('starting_street').T.to_dict('list')
    for (k, v) in street_heavy_map.items():
        street_heavy_map[k] = v[0]

    train['starting_street_heavy'] = train['starting_street'].map(street_heavy_map)
    test['starting_street_heavy'] = test['starting_street'].map(street_heavy_map)

    end_street_heavy = train.groupby(['end_street'])['trip_duration'].mean().reset_index()
    street_heavy_map = end_street_heavy.set_index('end_street').T.to_dict('list')
    for (k, v) in street_heavy_map.items():
        street_heavy_map[k] = v[0]

    train['end_street_heavy'] = train['end_street'].map(street_heavy_map)
    test['end_street_heavy'] = test['end_street'].map(street_heavy_map)

    # total_distance vs steps
    train['per_step_distance'] = train['total_distance'] / train['number_of_steps']
    test['per_step_distance'] = test['total_distance'] / test['number_of_steps']

    # total_travel_time vs steps
    train['per_travel_time'] = train['total_travel_time'] / train['number_of_steps']
    test['per_travel_time'] = test['total_travel_time'] / test['number_of_steps']


def main():
    train, test = data_utils.load_dataset(op_scope='5')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'load fastest routes dataset...'
    train_fr_1 = pd.read_csv('../input/fastest_routes_train_part_1.csv')
    train_fr_2 = pd.read_csv('../input/fastest_routes_train_part_2.csv')
    test_fr = pd.read_csv('../input/fastest_routes_test.csv')

    train_fr = pd.concat((train_fr_1, train_fr_2))

    train = train.merge(train_fr, how='left', on='id')
    test = test.merge(test_fr, how='left', on='id')

    print 'generate street heavy...'
    generate_street_heavy(train, test)

    train.drop(['starting_street', 'end_street', 'street_for_each_step',
                'distance_per_step', 'travel_time_per_step', 'step_maneuvers',
                'step_direction', 'step_location_list'], axis=1, inplace=True)
    test.drop(['starting_street', 'end_street', 'street_for_each_step',
               'distance_per_step', 'travel_time_per_step', 'step_maneuvers',
               'step_direction', 'step_location_list'], axis=1, inplace=True)

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='6')


if __name__ == '__main__':
    print '========== add fastest routes features =========='
    main()
