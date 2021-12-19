#!/bin/sh

python3 ../sanity_check/sanity_check_all.py      \
        --mode cascading \
        --dataset_path  /media/cornell/Data/tq42/Hyphal_2020/data   \
        --model_path  /media/cornell/Data/tq42/Hyphal_2020          \
        --model_type Inception3                                     \
        --pretrained                                                \
        --loading_epoch 191                                        \
        --cuda                                                      \
        --cuda_id 1                                                \
        --group weighted \
        --timestamp May11_05-52-52_2021                            