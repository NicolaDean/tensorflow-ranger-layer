#!/bin/bash

for freq in 0.25 0.5 0.75 1.0
do
    for epoch in 60
    do
        python post_fat_report.py --checkpoint MIXED_V3_${freq}_SINGLE_LAYER_batch_normalization_5 --experiment_name MIXED_V3_${freq} --epoch $epoch --layer batch_normalization_5
    done
done

for freq in 0.25 0.5 0.75 1.0
do
    for epoch in 60
    do
        python post_fat_report.py --checkpoint MIXED_V3_${freq}_SINGLE_LAYER_batch_normalization_25 --experiment_name MIXED_V3_${freq} --epoch $epoch --layer batch_normalization_25
    done
done

for freq in 0.25 0.5 0.75 1.0
do
    for epoch in 60
    do
        python post_fat_report.py --checkpoint MIXED_V3_${freq}_SINGLE_LAYER_conv2d_7 --experiment_name MIXED_V3_${freq} --epoch $epoch --layer conv2d_7
    done
done