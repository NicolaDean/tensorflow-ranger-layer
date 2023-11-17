#!/bin/bash
# Read a string with spaces using for loop

#MODELS
#faster_rcnn
#ssd-mobilenet
gen_report(){
    python post_fat_report_ssd.py --model_name ssd-mobilenet --experiment_name PREFAT_PEDESTRIAN --layer $1 --prefat --single_file --dataset pedestrian --dataset_path ./Self-Driving-Car-3
}

#LAUNCH EXPERIMENT ON A SPECIFIC LAYER AS FOLLOW
 #conv5_block2_2_conv (Conv2D)                                                            
                                                                                              
 #conv5_block2_2_bn (FreezableBa 
gen_report block_2_expand

. '''
for layer in 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16
    do
        gen_report block_${layer}_expand
        gen_report block_${layer}_project
        gen_report block_${layer}_expand_BN
        gen_report block_${layer}_depthwise_BN
        gen_report block_${layer}_project_BN
    done
