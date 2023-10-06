#!/bin/bash
# Read a string with spaces using for loop

train_all_model(){
    $LAYER=$1
    #PREFAT
    ARGS="--layer $1 --experiment_name PRE_FAT_SINGLE_LAYER --prefat"
    python post_fat_report.py $ARGS

    for freq in 1.0 0.5 0.75
        do              
            #VANILLA
            ARGS="--checkpoint FREQUENCY_${freq}_SINGLE_LAYER_$1 --experiment_name FREQUENCY_${freq} --epoch 20 --layer $1  --root ./results/$1"
            python post_fat_report.py $ARGS

            #GOLDEN
            ARGS="--checkpoint GOLDEN_FREQUENCY_${freq}_SINGLE_LAYER_$1 --experiment_name GOLDEN_${freq} --epoch 20 --layer $1  --root ./results/$1"
            python post_fat_report.py $ARGS

            #MIXED v1  alternate a livello di batch
            ARGS="--checkpoint MIXED_${freq}_SINGLE_LAYER_$1 --experiment_name MIXED_V1_${freq} --epoch 20 --layer $1  --root ./results/$1"
            python post_fat_report.py $ARGS

            #MIXED v3 alternate ma ricreiamo le label golden con i modello in maniera progressiva
            ARGS="--checkpoint MIXED_V3_${freq}_SINGLE_LAYER_$1 --experiment_name MIXED_V3_${freq} --epoch 60 --layer $1 --root ./results/$1"
            python post_fat_report.py $ARGS

        done
    
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW

#train_all_model "batch_normalization_2"
train_all_model "batch_normalization_3"
train_all_model "batch_normalization_4"
train_all_model "batch_normalization_5"
train_all_model "batch_normalization_6"
train_all_model "batch_normalization_7"
train_all_model "batch_normalization_8"
train_all_model "batch_normalization_9"
train_all_model "batch_normalization_20"
train_all_model "batch_normalization_30"
train_all_model "batch_normalization_40"
train_all_model "batch_normalization_50"
train_all_model "batch_normalization_60"
train_all_model "batch_normalization_70"

train_all_model "conv2d_3"
train_all_model "conv2d_5"
train_all_model "conv2d_6"
train_all_model "conv2d_7"
train_all_model "conv2d_8"
train_all_model "conv2d_9"
#train_all_model "conv2d_20"
#train_all_model "conv2d_30"
#train_all_model "conv2d_40"
#train_all_model "conv2d_50"
#train_all_model "conv2d_60"
#train_all_model "conv2d_70"
