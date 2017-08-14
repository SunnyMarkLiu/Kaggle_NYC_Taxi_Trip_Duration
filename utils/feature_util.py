#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-14 下午9:04
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd


def feature_check_before_modeling(train, test, check_feature_names):
    """
    It might save you some headache to check your train and test feature
    distributions before modeling. Usually in kaggle competitions train 
    and test sets are iid. If there is huge differenc between train and 
    test set than probably you have a bug in your feature extraction pipeline.
    """
    feature_stats = pd.DataFrame({'feature': check_feature_names})
    feature_stats.loc[:, 'train_mean'] = np.nanmean(train[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'test_mean'] = np.nanmean(test[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'train_std'] = np.nanstd(train[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'test_std'] = np.nanstd(test[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'train_nan'] = np.mean(np.isnan(train[check_feature_names].values), axis=0).round(3)
    feature_stats.loc[:, 'test_nan'] = np.mean(np.isnan(test[check_feature_names].values), axis=0).round(3)
    feature_stats.loc[:, 'train_test_mean_diff'] = np.abs(
        feature_stats['train_mean'] - feature_stats['test_mean']) / np.abs(
        feature_stats['train_std'] + feature_stats['test_std']) * 2
    feature_stats.loc[:, 'train_test_nan_diff'] = np.abs(feature_stats['train_nan'] - feature_stats['test_nan'])

    return feature_stats
