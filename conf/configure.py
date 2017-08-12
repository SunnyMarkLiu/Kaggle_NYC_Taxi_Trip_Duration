#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-26 下午3:14
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import time


class Configure(object):

    original_train_path = '../input/train.csv'
    original_test_path = '../input/test.csv'

    processed_train_path = '../input/operate_{}_train_dataset.pkl'
    processed_test_path = '../input/operate_{}_test_dataset.pkl'

    submission_path = '../result/submission_{}.csv'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))

    pikcup_time_window_cluster_traffic_features_path = '../input/pikcup_after_time_window_{}_cluster_{}_traffic_features.pkl'
    dropoff_time_window_cluster_traffic_features_path = '../input/dropoff_before_time_window_{}_cluster_{}_traffic_features.pkl'
