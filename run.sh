#!/usr/bin/env bash
cd features
sh run.sh
cd ../model/
python xgboost_model.py
