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


def load_dataset(op_scope):
    if not os.path.exists(Configure.processed_train_path.format(op_scope)):
        train = pd.read_csv(Configure.original_train_path)
    else:
        with open(Configure.processed_train_path.format(op_scope), "rb") as f:
            train = cPickle.load(f)

    if not os.path.exists(Configure.processed_test_path.format(op_scope)):
        test = pd.read_csv(Configure.original_test_path)
    else:
        with open(Configure.processed_test_path.format(op_scope), "rb") as f:
            test = cPickle.load(f)
    return train, test


def save_dataset(train, test, op_scope):
    if train is not None:
        with open(Configure.processed_train_path.format(op_scope), "wb") as f:
            cPickle.dump(train, f, -1)

    if test is not None:
        with open(Configure.processed_test_path.format(op_scope), "wb") as f:
            cPickle.dump(test, f, -1)
