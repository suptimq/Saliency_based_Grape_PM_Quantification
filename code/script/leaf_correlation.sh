#!/bin/bash

time python3 ../leaf_correlation.py \
                --model_type VGG                                            \
                --model_path /media/cornell/Data/tq42/Hyphal_2020           \
                --pretrained                                                \
                --dataset_path /media/cornell/Data/tq42/Hyphal_2020/data    \
                --loading_epoch 95                                         \
                --threshold 0.2 \
                --cuda                                                      \
                --cuda_id 0                                                 \
                --timestamp May10_16-42-59_2021