#!/bin/bash
# Read a string with spaces using for loop

script(){
    for freq in 1.0 0.25 0.5 0.75
        do  
            python single_layer_experiment.py --experiment_name CUSTOM_LOSS --layer $1 --frequency ${freq} --custom_loss
        done
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW

script "batch_normalization_5"
script "conv2d_7"
script "batch_normalization_25"
script "batch_normalization_9"
