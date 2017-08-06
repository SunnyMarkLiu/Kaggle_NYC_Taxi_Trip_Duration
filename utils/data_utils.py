#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-6 下午3:12
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
import cPickle
from conf.configure import Configure


def load_dataset():
    if not os.path.exists(Configure.processed_train_path):
        train = pd.read_csv(Configure.original_train_path)
    else:
        with open(Configure.processed_train_path, "rb") as f:
            train = cPickle.load(f)

    if not os.path.exists(Configure.processed_test_path):
        test = pd.read_csv(Configure.original_test_path)
    else:
        with open(Configure.processed_test_path, "rb") as f:
            test = cPickle.load(f)
    return train, test


def save_dataset(train, test):
    if train is not None:
        with open(Configure.processed_train_path, "wb") as f:
            cPickle.dump(train, f, -1)

    if test is not None:
        with open(Configure.processed_test_path, "wb") as f:
            cPickle.dump(test, f, -1)
