#!/bin/bash
# Read a string with spaces using for loop

report(){
    $LAYER=
    python post_fat_report.py  --layer $1 --experiment_name BOATS_DEV_PRE_FAT_SINGLE_LAYER --prefat 
    #--dataset ./Self-Driving-Car-3 --checkpoint ./results/test_final_v2.h5
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
report "conv2d_4"
report "conv2d_5"
report "conv2d_6"
report "conv2d_7"
report "conv2d_8"
report "conv2d_9"
report "conv2d_20"
report "conv2d_30"
report "conv2d_40"
report "conv2d_50"
report "conv2d_60"
report "conv2d_70"
