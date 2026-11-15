#!/bin/bash
curl -L --create-dirs -o ./src/data/heart-failure-prediction.zip\
  https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction
unzip ./src/data/heart-failure-prediction.zip -d ./src/data
