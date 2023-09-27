#!/bin/bash
# Read a string with spaces using for loop

train_all_model(){
    $LAYER=$1
    for freq in 1.0 0.5 0.75
        do  
            #VANILLA
            ARGS="--experiment_name FREQUENCY --layer $1 --frequency ${freq} --epochs 21"
            python single_layer_experiment.py $ARGS

            #GOLDEN
            ARGS="--experiment_name GOLDEN_FREQUENCY --layer $1 --frequency ${freq} --golden_label --epochs 21"
            python single_layer_experiment.py $ARGS

            #MIXED v1  alternate a livello di batch
            ARGS="--experiment_name MIXED --layer $1 --frequency ${freq} --mixed_label --epochs 21"
            python single_layer_experiment.py $ARGS

            #MIXED v2 alternate a livello di epoche
            #ARGS="--experiment_name MIXED_V2 --layer $1 --frequency ${freq} --mixed_label_v2 --num_epochs_switch 3"
            #python single_layer_experiment.py $ARGS

            #MIXED v3 alternate ma ricreiamo le label golden con i modello in maniera progressiva
            ARGS="--experiment_name MIXED_V3 --layer $1 --frequency ${freq} --mixed_label_v3 --num_epochs_switch 3 --epochs 61"
            python single_layer_experiment.py $ARGS

            #MIXED v4 usa ground truth training (0 injection) alternato con golden injection
            #ARGS="--experiment_name MIXED_V4 --layer $1 --frequency ${freq} --mixed_label_v4 --num_epochs_switch 3 --epochs 48"
            #python single_layer_experiment.py $ARGS

            #GOLDEN GT
            #ARGS="--experiment_name GOLDEN_GT --layer $1 --frequency ${freq} --golden_label --epochs 20 --golden_gt"
            #python single_layer_experiment.py $ARGS
        done
    
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW

#train_all_model "batch_normalization_2"
#train_all_model "batch_normalization_3"
#train_all_model "batch_normalization_4"
#train_all_model "batch_normalization_5"
#train_all_model "batch_normalization_6"
#train_all_model "batch_normalization_7"
#train_all_model "batch_normalization_8"
#train_all_model "batch_normalization_9"
#train_all_model "batch_normalization_20"
#train_all_model "batch_normalization_30"
#train_all_model "batch_normalization_40"
#train_all_model "batch_normalization_50"
#train_all_model "batch_normalization_60"
#train_all_model "batch_normalization_70"

#train_all_model "conv2d_2"
train_all_model "conv2d_3"
train_all_model "conv2d_4"
train_all_model "conv2d_5"
train_all_model "conv2d_6"
train_all_model "conv2d_7"
train_all_model "conv2d_8"
train_all_model "conv2d_9"
train_all_model "conv2d_20"
train_all_model "conv2d_30"
train_all_model "conv2d_40"
train_all_model "conv2d_50"
train_all_model "conv2d_60"
train_all_model "conv2d_70"
