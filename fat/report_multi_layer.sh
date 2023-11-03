#!/bin/bash

train_all_model(){
    python post_fat_report.py --layer $1 --epoch 130 --checkpoint $2/$2 --experiment_name $2 --single_f1_file
}


generate_report()
{   
    train_all_model "batch_normalization_3" $1
    train_all_model "batch_normalization_4" $1
    train_all_model "batch_normalization_5" $1
    train_all_model "batch_normalization_6" $1
    train_all_model "batch_normalization_7" $1
    train_all_model "batch_normalization_8" $1
    train_all_model "batch_normalization_9" $1
    train_all_model "batch_normalization_20" $1
    train_all_model "batch_normalization_30" $1
    train_all_model "batch_normalization_40" $1
    train_all_model "batch_normalization_50" $1
    train_all_model "batch_normalization_60" $1
    train_all_model "batch_normalization_70" $1
    train_all_model "conv2d_3" $1
    train_all_model "conv2d_4" $1
    train_all_model "conv2d_5" $1
    train_all_model "conv2d_6" $1
    train_all_model "conv2d_7" $1
    train_all_model "conv2d_8" $1
    train_all_model "conv2d_9" $1
    train_all_model "conv2d_20" $1
    train_all_model "conv2d_30" $1
    train_all_model "conv2d_40" $1
    train_all_model "conv2d_50" $1
    train_all_model "conv2d_60" $1
    train_all_model "conv2d_70" $1
    
}

generate_report MULTI_LAYER_FREQ_ZERO_0.5

#generate_report MULTI_LAYER_FREQ_MASK_GRAD_0.5
#generate_report MULTI_LAYER_FREQ_DEV_0.5
#generate_report MULTI_LAYER_FREQ_DEV_0.75
#generate_report MULTI_LAYER_MIXED_DEV_0.5
#generate_report MULTI_LAYER_MIXED_DEV_0.75


#generate_report MULTI_LAYER_MIXED_0.50
#generate_report MULTILAYER_MIXED_v3_0.5
#generate_report MULTILAYER_MIXED_v3_0.75
#generate_report NEW_CUSTOM_LOSS_MULTILAYER_1.0
#generate_report WEIGHTED_2_CUSTOM_LOSS_MULTILAYER_1.0