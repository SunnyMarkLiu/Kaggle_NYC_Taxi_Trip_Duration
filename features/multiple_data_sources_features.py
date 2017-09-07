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

import numpy as np
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


def add_weather_features(train, test):
    trip_durations = train['trip_duration']
    del train['trip_duration']
    conbined_data = pd.concat([train, test])

    weather_data = pd.read_csv('../input/weather_data_nyc_centralpark_2016.csv')
    weather_data['day'] = weather_data['date'].map(lambda d: d.split('-')[0])
    weather_data['month'] = weather_data['date'].map(lambda d: d.split('-')[1])

    weather_data['day'] = weather_data['day'].astype(np.uint8)
    weather_data['month'] = weather_data['month'].astype(np.uint8)

    df = pd.DataFrame({'year': [2016] * weather_data.shape[0],
                       'month': weather_data['month'],
                       'day': weather_data['day']})
    weather_data['pickup_date'] = pd.to_datetime(df)
    weather_data.drop(['date', 'day', 'month'], axis=1, inplace=True)
    weather_data.columns = ['maximum_temerature', 'minimum_temperature', 'average_temperature', 'precipitation',
                            'snow_fall', 'snow_depth', 'pickup_date']

    weather_data['precipitation'] = weather_data['precipitation'].map(lambda p: float(p) if p != 'T' else -1)
    weather_data['snow_fall'] = weather_data['snow_fall'].map(lambda p: float(p) if p != 'T' else -1)
    weather_data['snow_depth'] = weather_data['snow_depth'].map(lambda p: float(p) if p != 'T' else -1)

    weather_data.drop(['maximum_temerature', 'minimum_temperature',
                       'average_temperature', 'snow_depth', 'precipitation'], axis=1, inplace=True)

    conbined_data['pickup_datetime'] = pd.to_datetime(conbined_data['pickup_datetime'])
    conbined_data['pickup_date'] = conbined_data['pickup_datetime'].dt.date
    conbined_data['pickup_date'] = pd.to_datetime(conbined_data['pickup_date'])
    weather_data['pickup_date'] = pd.to_datetime(weather_data['pickup_date'])

    conbined_data = pd.merge(conbined_data, weather_data, on='pickup_date', how='left')

    del conbined_data['pickup_date']
    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    train['trip_duration'] = trip_durations

    return train, test

def main():
    train, test = data_utils.load_dataset(op_scope='5')
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'add fastest routes features...'
    train_fr_1 = pd.read_csv('../input/fastest_routes_train_part_1.csv')
    train_fr_2 = pd.read_csv('../input/fastest_routes_train_part_2.csv')
    test_fr = pd.read_csv('../input/fastest_routes_test.csv')

    train_fr = pd.concat((train_fr_1, train_fr_2))

    train = train.merge(train_fr, how='left', on='id')
    test = test.merge(test_fr, how='left', on='id')

    generate_street_heavy(train, test)

    train.drop(['starting_street', 'end_street', 'street_for_each_step',
                'distance_per_step', 'travel_time_per_step', 'step_maneuvers',
                'step_direction', 'step_location_list'], axis=1, inplace=True)
    test.drop(['starting_street', 'end_street', 'street_for_each_step',
               'distance_per_step', 'travel_time_per_step', 'step_maneuvers',
               'step_direction', 'step_location_list'], axis=1, inplace=True)

    print 'add weather features...'
    train, test = add_weather_features(train, test)
    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='6')


if __name__ == '__main__':
    print '========== add multiple_data_sources features =========='
    main()
