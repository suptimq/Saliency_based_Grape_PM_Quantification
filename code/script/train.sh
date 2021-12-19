#!/bin/bash

python3 ../classification/run.py      \
            --root_path  /media/cornell/Data/tq42/Hyphal_2020 \
            --model_type VGG            \
            --pretrained                \
            --weighted_loss            \
            --save_model                \
            --seg_dataset               \
            --seg_idx   $i              \
            --loading_epoch 0           \
            --total_epochs 300          \
            --cuda                      \
            --optimType Adam            \
            --lr 1e-4                  \
            --weight_decay 2e-4         \
            --bsize 100                 \
            --nworker 1                 \
            --cuda_device 1   