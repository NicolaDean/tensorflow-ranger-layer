from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys
import os
import pathlib
from keras.utils import img_to_array, array_to_img
from tqdm import tqdm

# directory reach
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *
from our_datasets import pet
from our_datasets import self_drive

#2 types of errors in object detection setting:
# 1) number of boxes predicted -> predicted number must be equal to real one (binary evaluation: EQUAL or NOT EQUAL)
# 2) correspondance 1 to 1 for each box -> match is done looking at max iou index (binary evaluation can be done setting a threshold for stating whether
# the boxes actually match or not)


F1_score_report = make_dataclass("F1_score_report",[("Layer_name",str),("NUM_ITER",int),("MISC_50",int),("MISC_60",int),("MISC_70",int),("ROB_50",float),("ROB_60",float),("ROB_70",float)])
    


def create_mask(pred_mask,idx=None):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    if idx == None:
        return pred_mask
    else:
        return pred_mask[idx]

def calculate_iou(gt_mask, pred_mask, class_check=1,threshold = True):

    #Create a mask of 1 where 
    pred_mask = tf.equal(pred_mask , class_check)
    gt_mask   = tf.equal(gt_mask , class_check)

    overlap = tf.cast(tf.math.logical_and(pred_mask,gt_mask),dtype="float32")
    union   = tf.cast(tf.math.logical_or(pred_mask,gt_mask),dtype="float32")

    tot_overlap = tf.math.reduce_sum(overlap)
    tot_union = tf.math.reduce_sum(union)
    
    if tot_union == 0 and tot_overlap == 0:
        return 1
    elif tot_union == 0:
        return 0
    
    iou =  tot_overlap / tot_union
    return iou


def compute_mean_iou(label,prediction,NUM_CLASSES):

    label       = create_mask(label)
    prediction  = create_mask(prediction)

    NUM_CLASSES = 32
    classes_iou = []
    for cls_idx in range(NUM_CLASSES):
        curr_iou = calculate_iou(prediction,label,cls_idx)
        #print(f"Class[{cls_idx}] => {curr_iou} ")
        classes_iou.append(curr_iou)
    
    mean_iou = sum(classes_iou) / NUM_CLASSES
    #print(f'MEAN [{mean_iou}]')

    return classes_iou, mean_iou.numpy()


def post_fat_segmentation_report(injection_point,out_prefix="PREFAT_NEW", prefat=False,SINGLE_F1_FILE=False,SKIP_INJECTION=False):


    OUTPUT_NAME    = f"./reports/unet/F1_{out_prefix}.csv"

    model = tf.keras.models.load_model("../saved_models/unet_self_drive_v2")

    model.load_weights("../saved_models/unet_FAT_test.h5",by_name=True)
    
    #model = tf.keras.models.load_model("../saved_models/unet_FAT_test")

    model.summary()

    train_generator_fn,place_holder = self_drive.get_generator(batch_size=32,not_valid=True)
    place_holder,val_generator_fn   = self_drive.get_generator(batch_size=1,not_train=True)

    train_size = 369
    valid_size = 100

    train_generator_fn = train_generator_fn()
    val_generator_fn   = val_generator_fn()

    #RAGE TUNE THE YOLO MODEL
    def range_tune(RANGER):
        print("=============FINE TUNING=============")
        if not SKIP_INJECTION:
            for idx in tqdm(range(int(train_size//32))):
                x_t,y_t = next(train_generator_fn)
                RANGER.tune_model_range(x_t, reset=False,verbose=False)

    print("Layers on which we inject faults: ", str(injection_point))
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(model,[injection_point],NUM_INJECTIONS=60,use_classes_ranging=True,range_tuning_fn=range_tune,verbose=True)
    inj_model = RANGER.get_model()
    #yolo_ranger.summary()
    CLASSES.set_model(inj_model)
    CLASSES.disable_all(verbose=False)
    #RANGER.set_ranger_mode(mode=RangerModes.RangeTuning)
    
    inj_model.summary()
    
    ########################## REPORT #########################

    #Inizialize experiment variables
    print("-------------------------------")
    print(f'Injection on layer {injection_point}')
    print("-------------------------------")



    layer = CLASSES_HELPER.get_layer(inj_model,"classes_" + injection_point,verbose=False)
    assert isinstance(layer, ErrorSimulator)
    layer.set_mode(ErrorSimulatorMode.enabled)

    idx = 0

    progress_bar = tqdm(range(valid_size))
    
    tot_gold_iou = 0
    tot_inj_iou  = 0
    tot_gt_iou  = 0
    MISC_50 = 0
    MISC_60 = 0
    MISC_70 = 0
    
    count = 0
    inj_batch_size = 64
    report=[]
    for sample_id in progress_bar:
        
        img,label = next(val_generator_fn)

        #img   = [img[0]] * inj_batch_size
        #label = [label[0]] * inj_batch_size

        #img     = np.stack(img)
        #label   = np.stack(label)

        golden_pred     = model.predict(img,verbose=False)
        gold_classes_iou, gold_mean_iou = compute_mean_iou(label,golden_pred,NUM_CLASSES=32)

        for _ in range(inj_batch_size):
            injection_pred  = inj_model.predict(img,verbose=False)
            
            
            inj_classes_iou,  inj_mean_iou  = compute_mean_iou(golden_pred,injection_pred,NUM_CLASSES=32)
            gt_classes_iou,  gt_mean_iou    = compute_mean_iou(label,injection_pred,NUM_CLASSES=32)
            
            inj_classes_iou = np.array(inj_classes_iou)

            if sum(inj_classes_iou < 0.5) > 0:
                MISC_50 += 1
            if sum(inj_classes_iou < 0.6) > 0:
                MISC_60 += 1
            if sum(inj_classes_iou < 0.7) > 0:
                MISC_70 += 1
            #print(inj_classes_iou)

            count += 1
            tot_gold_iou += gold_mean_iou
            tot_inj_iou  += inj_mean_iou
            tot_gt_iou   += gt_mean_iou
            
            mean_tot_gold_iou   = tot_gold_iou / count
            mean_tot_inj_iou    = tot_inj_iou / count
            mean_tot_gt_iou     = tot_gt_iou / count

            ROB_50 = 1-MISC_50/count
            ROB_60 = 1-MISC_60/count
            ROB_70 = 1-MISC_70/count
            
            progress_bar.set_postfix({'ROB_50': ROB_50,'ROB_60': ROB_60,'ROB_70':ROB_70})

    report = [F1_score_report(injection_point,count,MISC_50,MISC_60,MISC_70,ROB_50,ROB_60,ROB_70)]
    f1_score_report = pd.DataFrame(report)
    f1_score_report.to_csv(OUTPUT_NAME, mode = 'a', header = False, decimal = ',', sep=';')

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer"      , action = "store",default="conv2d_4")   

    args            = parser.parse_args()
    layer          = args.layer

    post_fat_segmentation_report(injection_point=layer,out_prefix="FREQ_50")