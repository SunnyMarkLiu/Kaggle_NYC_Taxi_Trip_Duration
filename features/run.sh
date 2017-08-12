#!/usr/bin/env bash

python train_test_preprocess.py
python basic_feature_engineering.py
python data_cleaning.py
python perform_geography_clustering.py
python time_feature_engineering.py
