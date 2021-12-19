#!/bin/bash

time python3 ../patch_correlation.py \
		 --model_type VGG                                            \
		 --model_path /media/cornell/Data/tq42/Hyphal_2020           \
         --pretrained                                                \
		 --dataset_path /media/cornell/Data/tq42/Hyphal_2020/data    \
		 --loading_epoch 95                                         \
         --cuda                                                      \
         --cuda_id 1                                                 \
         --threshold 0.2   \
		 --timestamp May10_16-42-59_2021
