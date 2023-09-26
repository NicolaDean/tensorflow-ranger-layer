#!/bin/bash
# Read a string with spaces using for loop

script(){

    
    for freq in 1.0 0.25 0.5 0.75
        do  
            #GOLDEN
            python single_layer_experiment.py --experiment_name GOLDEN_GT --layer $1 --frequency ${freq} --golden_label --epochs 20 --golden_gt
        done
    
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW

script "batch_normalization_5"
script "conv2d_7"
script "batch_normalization_25"
script "batch_normalization_9"
