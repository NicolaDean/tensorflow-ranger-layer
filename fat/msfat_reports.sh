#!/bin/bash

train_all_model(){
    python post_fat_report.py --layer $1 --epoch 5 --checkpoint $2 --experiment_name $3 --single_f1_file --force_checkpoint_name --force_out_name $4 --num_iteration $5
}


generate_report()
{   
    train_all_model "batch_normalization_3" $1 $2 $3 $4
    train_all_model "batch_normalization_4" $1 $2 $3 $4
    train_all_model "batch_normalization_5" $1 $2 $3 $4
    train_all_model "batch_normalization_6" $1 $2 $3 $4
    train_all_model "batch_normalization_7" $1 $2 $3 $4
    train_all_model "batch_normalization_8" $1 $2 $3 $4
    train_all_model "batch_normalization_9" $1 $2 $3 $4
    train_all_model "batch_normalization_20" $1 $2 $3 $4
    train_all_model "batch_normalization_30" $1 $2 $3 $4
    train_all_model "batch_normalization_40" $1 $2 $3 $4
    train_all_model "batch_normalization_50" $1 $2 $3 $4
    train_all_model "batch_normalization_60" $1 $2 $3 $4
    train_all_model "batch_normalization_70" $1 $2 $3 $4
    train_all_model "conv2d_3" $1 $2 $3 $4
    train_all_model "conv2d_4" $1 $2 $3 $4
    train_all_model "conv2d_5" $1 $2 $3 $4
    train_all_model "conv2d_6" $1 $2 $3 $4
    train_all_model "conv2d_7" $1 $2 $3 $4
    train_all_model "conv2d_8" $1 $2 $3 $4
    train_all_model "conv2d_9" $1 $2 $3 $4
    train_all_model "conv2d_20" $1 $2 $3 $4
    train_all_model "conv2d_30" $1 $2 $3 $4
    train_all_model "conv2d_40" $1 $2 $3 $4
    train_all_model "conv2d_50" $1 $2 $3 $4
    train_all_model "conv2d_60" $1 $2 $3 $4
    train_all_model "conv2d_70" $1 $2 $3 $4
}

generate_report $2 "NULL_MSFAT" $1  $3

#'--dataset ./Self-Driving-Car-3'