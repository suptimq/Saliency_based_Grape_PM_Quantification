#!/bin/bash


python3 ../classification/inference.py      \
            --dataset_path  /media/cornell/Data/tq42/Hyphal_2020   \
            --model_path  /media/cornell/Data/tq42/Hyphal_2020   \
            --model_type VGG                         \
            --pretrained \
            --set val \
            --loading_epoch 147                              \
            --test_date 07-11-19_9dpi \
            --group weighted \
            --cuda                                          \
            --cuda_id 0                                      \
            --timestamp May11_00-40-16_2021

            