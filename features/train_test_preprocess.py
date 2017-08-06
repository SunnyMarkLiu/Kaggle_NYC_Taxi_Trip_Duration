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

def main():
    train, test = data_utils.load_dataset()
    print 'train: {}, test: {}'.format(train.shape, test.shape)

if __name__ == '__main__':
    print '========== perform simple train test preprocess =========='
    main()