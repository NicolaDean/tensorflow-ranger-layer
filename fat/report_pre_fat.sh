#!/bin/bash
# Read a string with spaces using for loop

report(){
    $LAYER=
    python post_fat_report.py --checkpoint ./results/test_final_v2.h5 --layer $1 --experiment_name PEDESTRIAN_PRE_FAT_SINGLE_LAYER --prefat --dataset ./Self-Driving-Car-3
}

#report batch_normalization_5
report "batch_normalization_3"
report "batch_normalization_4"
report "batch_normalization_5"
report "batch_normalization_6"
report "batch_normalization_7"
report "batch_normalization_8"
report "batch_normalization_9"
report "batch_normalization_20"
report "batch_normalization_30"
report "batch_normalization_40"
report "batch_normalization_50"
report "batch_normalization_60"
report "batch_normalization_70"

report "conv2d_3"
report "conv2d_5"
report "conv2d_6"
report "conv2d_7"
report "conv2d_8"
report "conv2d_9"



