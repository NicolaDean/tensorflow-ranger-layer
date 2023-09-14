#!/bin/bash

for freq in 0.25 0.5 0.75 1.0
do
    for epoch in 20 30
    do
        python post_fat_report.py --checkpoint batch_normalization_5/MIXED_${freq}_SINGLE_LAYER_batch_normalization_5 --experiment_name MIXED_${freq} --epoch $epoch --layer batch_normalization_5
    done
done