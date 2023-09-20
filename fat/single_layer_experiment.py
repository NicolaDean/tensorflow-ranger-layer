from utils.fat_experiment import *
import sys
import argparse
import os
from tqdm import tqdm

'''
injection_points = ["conv2d", "batch_normalization"]
injection_points += ["conv2d_"+str(i) for i in range(1, 10)]
injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]
'''

#injection_points = ["conv2d"]
#injection_points = ["batch_normalization"]
#injection_points = ["conv2d_"+str(i) for i in range(1, 10)]
#injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
injection_points += ["batch_normalization_"+str(i) for i in range(5, 10)]
injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name" , default="Generic_experiment", action = "store")
    parser.add_argument("--golden_label"    , default=False, action='store_true')
    parser.add_argument("--mixed_label"     , default=False, action='store_true')
    parser.add_argument("--mixed_label_v2"  , default=False, action='store_true')
    parser.add_argument("--mixed_label_v3"  , default=False, action='store_true')
    parser.add_argument("--custom_loss"     , default=False, action='store_true')
    parser.add_argument("--frequency"       , default=0.5  , action = "store")
    parser.add_argument("--epochs"          , default=36   , action = "store")
    parser.add_argument("--layer"           , default="conv2d_5"   , action = "store")
    parser.add_argument("--switch_prob"     , default =0.5      , action = "store")
    parser.add_argument("--num_epochs_switch", default =1     , action = "store")

    args            = parser.parse_args()
    prefix          = args.golden_label
    epoch           = str(args.epochs)
    experiment_name = str(args.experiment_name)

    LAYER               = str (args.layer)
    EXPERIMENT_NAME     = f"{experiment_name}_{args.frequency}_SINGLE_LAYER_" + LAYER
    FINAL_WEIGHT_NAME   = f"single_layer_{LAYER}_final.h5"
    EPOCHS              = int(args.epochs)
    INJECTION_FREQUENCY = float(args.frequency)
    GOLDEN_LABEL        = bool(args.golden_label)
    MIXED_LABEL         = bool(args.mixed_label)
    MIXED_LABEL_V2      = bool(args.mixed_label_v2)
    MIXED_LABEL_V3      = bool(args.mixed_label_v3)
    SWITCH_PROB         = float(args.switch_prob)
    NUM_EPOCHS_SWITCH   = int(args.num_epochs_switch)
    CUSTOM_LOSS         = bool(args.custom_loss)

    injection_points  = [LAYER]
    print(f"Train for layer: {LAYER}")
    run_fat_experiment(EPOCHS,EXPERIMENT_NAME,FINAL_WEIGHT_NAME,injection_points = injection_points,GOLDEN_LABEL=GOLDEN_LABEL,MIXED_LABEL=MIXED_LABEL, MIXED_LABEL_V2=MIXED_LABEL_V2,
                        MIXED_LABEL_V3=MIXED_LABEL_V3, injection_frequency=INJECTION_FREQUENCY, switch_prob=SWITCH_PROB,num_epochs_switch = NUM_EPOCHS_SWITCH)
    
