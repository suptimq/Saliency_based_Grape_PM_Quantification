#!/bin/bash

#declare -a h=("hello" "world")
#declare -a code_paths=("`/Users/tim/BB_analysis/code/classification/run.py`" "`/home/tq42/BB_analysis/code/classification/run.py`")
#declare -a root_paths=("`/Users/tim/Documents/Cornell/CAIR/BlackBird/Data/Hyphal_2019`" "`/mnt/cornell/Data/tq42/Hyphal_2020`")

for((i=5;i<10;i++))
do
    python3 ../segmentation/run.py      \
                --root_path  /media/cornell/Data/tq42/Hyphal_2020 \
                --model_type DeepLab            \
                --pretrained                \
                --save_model                \
                --weighted_loss            \
                --loading_epoch 0           \
                --total_epochs 100          \
                --cuda                      \
                --cv                        \
                --seg_idx $i                \
                --optimType Adam            \
                --lr 1e-4                   \
                --weight_decay 2e-4         \
                --bsize 32                 \
                --nworker 1                 \
                --cuda_device 1             
done