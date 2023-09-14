#!/bin/bash
# Read a string with spaces using for loop

script(){
    for freq in 1.0 0.25 0.5 0.75
        do  
            
            python single_layer_experiment.py --experiment_name MIXED_V2 --layer $1 --frequency ${freq} --mixed_label_v2 --num_epochs_switch 3
        done
    
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW

script "batch_normalization_5"
script "conv2d_7"
script "batch_normalization_25"
