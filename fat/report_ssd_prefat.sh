#!/bin/bash
# Read a string with spaces using for loop

gen_report(){
    python post_fat_report_ssd.py --model_name ssd --experiment_name PREFAT --layer $1 
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW

for layer in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16
    do
        gen_report block_${layer}_expand
    done
