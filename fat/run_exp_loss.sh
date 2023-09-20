#!/bin/bash
# Read a string with spaces using for loop

conda activate nicola

#python post_fat_report.py --checkpoint ../../../keras-yolo3/yolo_boats_final.h5 --experiment_name PRE_FAT --epoch 200


gen_report(){
    LAYER=$1
    
    for freq in 1.0 0.25 0.5 0.75
    do
        for epoch in 10 20 30
        do
            python post_fat_report.py  --checkpoint CUSTOM_LOSS_${freq}_SINGLE_LAYER_$LAYER --experiment_name CUSTOM_LOSS_${freq}_SINGLE_LAYER --epoch $epoch --layer $LAYER --root ./results/$LAYER 
        done
    done
    
}

gen_report batch_normalization_5
gen_report conv2d_7

#gen_report batch_normalization_9



