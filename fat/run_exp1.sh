#!/bin/bash
# Read a string with spaces using for loop

conda activate nicola

#python post_fat_report.py --checkpoint ../../../keras-yolo3/yolo_boats_final.h5 --experiment_name PRE_FAT --epoch 200
$LAYER = "batch_normalization_5"

for freq in 1.0 0.25 0.5 0.75
    do
    for epoch in 10 20 30
        do
        python post_fat_report.py --layer $LAYER --checkpoint FREQUENCY_${freq}__SINGLE_LAYER_$LAYER --experiment_name FREQ_${freq}_SINGLE_LAYER --epoch $epoch 
        done
    done

for freq in 1.0 0.25 0.5 0.75
    do
    for epoch in 10 20 30
        do
        python post_fat_report.py --layer $LAYER --checkpoint GOLDEN_FREQUENCY_${freq}_SINGLE_LAYER_$LAYER --experiment_name GOLDEN_FREQ_${freq}_SINGLE_LAYER --epoch $epoch 
        done
    done





