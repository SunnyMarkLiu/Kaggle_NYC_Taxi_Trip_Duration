#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-6 下午3:11
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from utils import data_utils
from conf.configure import Configure


def main():
    if os.path.exists(Configure.processed_train_path.format('0')):
        return

    train, test = data_utils.load_dataset(op_scope='0')
    print 'train: {}, test: {}'.format(train.shape, test.shape)

    # store_and_fwd_flag
    train['is_store_and_fwd_flag'] = train['store_and_fwd_flag'].map(lambda s: 1 if s == 'Y' else 0)
    test['is_store_and_fwd_flag'] = test['store_and_fwd_flag'].map(lambda s: 1 if s == 'Y' else 0)
    del train['store_and_fwd_flag']
    del test['store_and_fwd_flag']

    print 'train: {}, test: {}'.format(train.shape, test.shape)
    print 'save dataset...'
    data_utils.save_dataset(train, test, op_scope='0')


if __name__ == '__main__':
    print '========== perform simple train test preprocess =========='
    main()
