#!/bin/bash
# Read a string with spaces using for loop

report(){
    $LAYER=
    python post_fat_report.py --layer $1 --experiment_name PRE_FAT_SINGLE_LAYER_$1 --prefat
}

#report batch_normalization_5
report conv2d_7
report batch_normalization_25
report batch_normalization_9



