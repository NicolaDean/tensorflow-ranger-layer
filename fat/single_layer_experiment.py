import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) != 0:
    print("LIMIT GPU GROWTH")
    tf.config.experimental.set_memory_growth(gpu[0], True) #limits gpu memory

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
injection_points =  []
injection_points += ["batch_normalization_"+str(i) for i in range(3, 10)]
injection_points += ["batch_normalization_"+str(i*10) for i in range(2, 8)]
injection_points += ["conv2d_"+str(i) for i in range(3, 10)]
injection_points += ["conv2d_"+str(i*10) for i in range(2, 8)]

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name" , default="Generic_experiment", action = "store")
parser.add_argument("--golden_label"    , default=False, action='store_true')
parser.add_argument("--mixed_label"     , default=False, action='store_true')
parser.add_argument("--mixed_label_v2"  , default=False, action='store_true')
parser.add_argument("--mixed_label_v3"  , default=False, action='store_true')
parser.add_argument("--mixed_label_v4"  , default=False, action='store_true')
parser.add_argument("--golden_gt"       , default=False, action='store_true')
parser.add_argument("--custom_loss"     , default=False, action='store_true')
parser.add_argument("--custom_loss_v2"  , default=False, action='store_true')
parser.add_argument("--frequency"       , default=0.5  , action = "store")
parser.add_argument("--epochs"          , default=36   , action = "store")
parser.add_argument("--layer"           , default="conv2d_5"   , action = "store")
parser.add_argument("--switch_prob"     , default =0.5      , action = "store")
parser.add_argument("--num_epochs_switch", default =1     , action = "store")
parser.add_argument("--multi_layer"     , default=False, action='store_true')
parser.add_argument("--unif_stack_policy", default=False,action='store_true')
parser.add_argument("--vanilla_training",default=False,action='store_true')
parser.add_argument("--dataset_path"    ,default="./../../keras-yolo3/",action='store')
parser.add_argument("--init_model"      ,default="./../../keras-yolo3/yolo_boats_final.h5",action='store')
parser.add_argument("--extraction_type" ,default=1,action='store')
parser.add_argument("--loss_w",default=1,action='store')
parser.add_argument("--msfat" ,default=-1,action='store')
STARTING_POINT_REPORT   = "./reports/yolo/F1_REPORT_BOATS_DEV_PRE_FAT_SINGLE_LAYER.csv"
parser.add_argument("--starting_point_report"       , default=STARTING_POINT_REPORT, action='store')

def fat_experiment(injection_points = injection_points,args = parser.parse_args()):

    prefix          = args.golden_label
    epoch           = str(args.epochs)
    experiment_name = str(args.experiment_name)

    LAYER               = str (args.layer)
    FINAL_WEIGHT_NAME   = f"single_layer_{LAYER}_{args.frequency}_final.h5"
    EPOCHS              = int(args.epochs)
    INJECTION_FREQUENCY = float(args.frequency)
    GOLDEN_LABEL        = bool(args.golden_label)
    MIXED_LABEL         = bool(args.mixed_label)
    MIXED_LABEL_V2      = bool(args.mixed_label_v2)
    MIXED_LABEL_V3      = bool(args.mixed_label_v3)
    MIXED_LABEL_V4      = bool(args.mixed_label_v4)
    GOLDEN_GT           = bool(args.golden_gt)
    SWITCH_PROB         = float(args.switch_prob)
    NUM_EPOCHS_SWITCH   = int(args.num_epochs_switch)
    CUSTOM_LOSS         = bool(args.custom_loss)
    CUSTOM_LOSS_V2      = bool(args.custom_loss_v2)
    MULTI_LAYERS_FLAG   = bool(args.multi_layer)
    UNIFORM_LAYER_POLICY= bool(args.unif_stack_policy)
    VANILLA_TRAINING    = bool(args.vanilla_training)
    DATASET             = str(args.dataset_path)
    WEIGHT_FILE_PATH    = str(args.init_model)
    LOSS_W              = float(args.loss_w)
    extraction_type     = int(args.extraction_type)
    MSFAT               = int(args.msfat)
    
    
    if extraction_type >= 2:
        injection_points = []
        for i in range(3,10):
            injection_points += ["conv2d_"+str(i)]
            injection_points += ["batch_normalization_"+str(i)]

        for i in range(2,8):
            injection_points += ["conv2d_"+str(i*10)]
            injection_points += ["batch_normalization_"+str(i*10)]

        EPOCHS = len(injection_points)*5 + 1

    if not MULTI_LAYERS_FLAG:
        injection_points    = [LAYER]
        EXPERIMENT_NAME     = f"{experiment_name}_{args.frequency}_SINGLE_LAYER_" + LAYER
        print(f"Train for layer: {LAYER}")
    elif MSFAT != -1:
        EXPERIMENT_NAME = experiment_name
    else:
        EXPERIMENT_NAME = f"{experiment_name}_{args.frequency}"


    run_fat_experiment(EPOCHS,
                       EXPERIMENT_NAME,
                       FINAL_WEIGHT_NAME,
                       injection_points = injection_points,
                       GOLDEN_LABEL=GOLDEN_LABEL,
                       MIXED_LABEL=MIXED_LABEL,
                       MIXED_LABEL_V2=MIXED_LABEL_V2,
                       MIXED_LABEL_V3=MIXED_LABEL_V3,
                       MIXED_LABEL_V4=MIXED_LABEL_V4,
                       GOLDEN_GT=GOLDEN_GT,
                       injection_frequency=INJECTION_FREQUENCY,
                       switch_prob=SWITCH_PROB,
                       num_epochs_switch = NUM_EPOCHS_SWITCH,
                       custom_loss=CUSTOM_LOSS,
                       custom_loss_v2=CUSTOM_LOSS_V2,
                       MULTI_LAYERS_FLAG = MULTI_LAYERS_FLAG,
                       UNIFORM_LAYER_POLICY=UNIFORM_LAYER_POLICY,
                       DATASET=DATASET,
                       VANILLA_TRAINING=VANILLA_TRAINING,
                       WEIGHT_FILE_PATH=WEIGHT_FILE_PATH,
                       extraction_type=extraction_type,
                       MSFAT=MSFAT,
                       LOSS_W=LOSS_W)
    
if __name__ == '__main__':
    fat_experiment()