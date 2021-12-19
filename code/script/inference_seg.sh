#!/bin/bash

declare -a times=("Aug14_11-22-25_2021" "Aug14_12-41-40_2021" "Aug14_14-00-59_2021" "Aug14_15-20-04_2021" "Aug14_16-39-26_2021" "Aug14_11-22-36_2021" "Aug14_12-42-37_2021" "Aug14_14-02-49_2021" "Aug14_15-22-59_2021" "Aug14_16-42-52_2021")
declare -a eps=("94" "100" "85" "86" "98" "100" "90" "100" "96" "95")

len=${#times[@]}


# python3 ../classification/inference.py                               \
#             --dataset_path  /media/cornell/Data/tq42/Hyphal_2020   \
#             --model_path  /media/cornell/Data/tq42/Hyphal_2020     \
#             --model_type DeepLab                                   \
#             --pretrained                                           \
#             --set val                                              \
#             --loading_epoch 94                                    \
#             --group weighted                                       \
#             --thicken                                              \
#             --cuda                                                 \
#             --seg_idx  0                                          \
#             --cuda_id 0                                            \
#             --timestamp Aug14_11-22-25_2021


for((i=0;i<10;i++))
do
    time=${times[i]}
    ep=${eps[i]}
    python3 ../segmentation/inference.py                               \
                --dataset_path  /media/cornell/Data/tq42/Hyphal_2020   \
                --model_path  /media/cornell/Data/tq42/Hyphal_2020     \
                --model_type DeepLab                                   \
                --pretrained                                           \
                --set val                                              \
                --loading_epoch $ep                                    \
                --group weighted                                       \
                --cuda                                                 \
                --seg_idx  $i                                          \
                --cuda_id 0                                            \
                --timestamp $time


    # time python3 ../analyzer_vs_manual_pixel_seg.py \
    #                 --model_type DeepLab                                            \
    #                 --model_path /media/cornell/Data/tq42/Hyphal_2020               \
    #                 --pretrained                                                    \
    #                 --dataset_path /media/cornell/Data/tq42/Hyphal_2020/data        \
    #                 --loading_epoch $ep                                              \
    #                 --threshold 0.2                                                 \
    #                 --cuda                                                          \
    #                 --cuda_id 0                                                     \
    #                 --seg_idx  $i                                                   \
    #                 --group seg                                                     \
    #                 --timestamp $time
done