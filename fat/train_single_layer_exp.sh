#!/bin/bash
# Read a string with spaces using for loop

script(){

    
    for freq in 1.0 0.25 0.5 0.75
        do  
            #VANILLA
            python single_layer_experiment.py --experiment_name FREQUENCY --layer $1 --frequency ${freq}
            #GOLDEN
            python single_layer_experiment.py --experiment_name GOLDEN_FREQUENCY --layer $1 --frequency ${freq} --golden_label
        done
    
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW

script "conv2d_7"
script "batch_normalization_25"
script "batch_normalization_9"
