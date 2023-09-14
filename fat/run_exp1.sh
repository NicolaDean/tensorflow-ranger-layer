#!/bin/bash
# Read a string with spaces using for loop

conda activate nicola

#python post_fat_report.py --checkpoint ../../../keras-yolo3/yolo_boats_final.h5 --experiment_name PRE_FAT --epoch 200


gen_report(){
    LAYER=$1
    
    for freq in 1.0 0.25 0.5 0.75
    do
        for epoch in 20
        do
            if [ "$2" == "NO_VANILLA"]
                then
                    echo "NO VANILLA"
                else
                    python post_fat_report.py  --checkpoint FREQUENCY_${freq}_SINGLE_LAYER_$LAYER --experiment_name FREQ_${freq}_SINGLE_LAYER --epoch $epoch --layer $LAYER --root ./results/$LAYER 
            fi
        done
    done
    
    for freq in 1.0 0.25 0.5 0.75
    do
        for epoch in 20
        do
            if [ "$2" == "NO_GOLDEN"]
                then
                    echo "NO_GOLDEN"
                else
                    python post_fat_report.py --checkpoint GOLDEN_FREQUENCY_${freq}_SINGLE_LAYER_$LAYER --experiment_name GOLDEN_FREQ_${freq}_SINGLE_LAYER --epoch $epoch --layer $LAYER --root ./results/$LAYER 
            fi
        done
    done
}

#gen_report conv2d_7 NO_VANILLA
gen_report batch_normalization_25
gen_report batch_normalization_9



