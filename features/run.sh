#!/usr/bin/env bash

python train_test_preprocess.py
python basic_feature_engineering.py
python data_cleaning.py
python perform_geography_clustering.py
python drop_some_features.py
#python generate_time_window_features.py    --> useless
#python generate_heavy_traffic_distance.py  --> too many features, overfitting!
python other_feature_engineering.py
python multiple_data_sources_features.py
python final_feature_engineering.py
