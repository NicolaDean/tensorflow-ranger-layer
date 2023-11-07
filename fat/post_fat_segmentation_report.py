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
from datasets import pet

#2 types of errors in object detection setting:
# 1) number of boxes predicted -> predicted number must be equal to real one (binary evaluation: EQUAL or NOT EQUAL)
# 2) correspondance 1 to 1 for each box -> match is done looking at max iou index (binary evaluation can be done setting a threshold for stating whether
# the boxes actually match or not)

Error_ID_report = make_dataclass("Error_ID_report", 
                                     [("Set", str), ("Layer_name", str), ("Sample_id", int), ("Cardinality", int), ("Pattern", int), 
                                      ("Golden_num_boxes", int), ("Faulty_num_boxes", int), 
                                      ("Precision", float), ("Recall", float), ("F1_score", float),
                                      ("True_positives", float), ("False_positives", float), ("False_negatives", float), ("Error", str)])
#("Num_excluded",int),("Num_empty")
F1_score_report = make_dataclass("F1_score_report",[("Layer_name",str),("Epoch",int),("Num_wrong_box_shape",int),("Num_wrong_box_count",int),("TOT_Num_Misclassification",int),
                                                        ("Robustness",float),("V_F1_score",float),("I_F1_score",float),("G_F1_score",float),("V_accuracy",float),("I_accuracy",float),("G_accuracy",float),
                                                        ("V_precision",float),("I_precision",float),("G_precision",float),("V_recall",float),("I_recall",float),("G_recall",float),])
    


def create_mask(pred_mask,idx=None):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    if idx == None:
        return pred_mask
    else:
        return pred_mask[idx]

def calculate_iou(gt_mask, pred_mask, class_check=1,threshold = True):

    pred_mask = tf.equal(pred_mask , class_check)
    gt_mask   = tf.equal(gt_mask , class_check)

    overlap = tf.cast(tf.math.logical_and(pred_mask,gt_mask),dtype="float32")
    union   = tf.cast(tf.math.logical_or(pred_mask,gt_mask),dtype="float32")

    iou = tf.math.reduce_sum(overlap) / tf.math.reduce_sum(union)
    return iou

def post_fat_segmentation_report(injection_point,out_prefix="test", prefat=False,SINGLE_F1_FILE=False,SKIP_INJECTION=False):


    OUTPUT_NAME    = f"./reports/unet/{injection_point}/{out_prefix}_epoch.csv"
    OUTPUT_NAME_F1 = f"./reports/unet/{injection_point}/F1_REPORT_{out_prefix}_{injection_point}.csv"
    
    if SINGLE_F1_FILE:
        OUTPUT_NAME_F1 = f"./reports/unet/F1_REPORT_{out_prefix}.csv" 

    
    NUM_ITERATATION_PER_SAMPLE = 50


    model = tf.keras.models.load_model("../saved_models/unet_pet")
    

    #RAGE TUNE THE YOLO MODEL
    def range_tune(RANGER):
        print("=============FINE TUNING=============")
        if not SKIP_INJECTION:
            for image, mask in tqdm(train_batches.take(200)):
                #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
                RANGER.tune_model_range(image, reset=False)

    print("Layers on which we inject faults: ", str(injection_point))
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(model,[injection_point],NUM_INJECTIONS=60,range)
    inj_model = RANGER.get_model()
    #yolo_ranger.summary()
    CLASSES.set_model(inj_model)
    CLASSES.disable_all(verbose=False)
    
    train_batches,validation_batches,test_batches,train_size,valid_size = pet.load_train(BATCH_SIZE=1)

    print(f"SIZE = [{train_size}]")


    ########################## REPORT #########################


    #Inizialize experiment variables
    print("-------------------------------")
    print(f'Injection on layer {injection_point}')
    print("-------------------------------")
    num_misclassification_box_shape = 0
    num_misclassification_wrong_box = 0
    num_misclassification           = 0
    num_of_injection_comleted       = 0
    robustness                      = 0
    num_excluded                    = 0
    num_empty                       = 0

    V_TP,V_FP,V_FN = 0,0,0
    I_TP,I_FP,I_FN = 0,0,0
    
    progress_bar = tqdm(range(valid_size))
    report = []


    print(type(validation_batches))
    for image, mask in validation_batches.take(2) :
        
        #Vanilla Model
        CLASSES.disable_all(verbose=False)
        RANGER.set_ranger_mode(RangerModes.Disabled)
        gold_predict = inj_model.predict(image)

        idx = 0
        gold_predict = create_mask(gold_predict,idx)
        
        iou_0 = calculate_iou(mask[idx],gold_predict,0)
        iou_1 = calculate_iou(mask[idx],gold_predict,1)
        iou_2 = calculate_iou(mask[idx],gold_predict,2)

        mean  = (iou_0 + iou_1 + iou_2) / 3 
    
        print(f'GOLDEN IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}]')
        #pet.display([image[0], mask[0], create_mask(pred_mask,0)])
        #INJECTION:
        CLASSES.disable_all(verbose=False)
        #layer = CLASSES_HELPER.get_layer(inj_model,"classes_" + injection_point,verbose=False)
        #assert isinstance(layer, ErrorSimulator)
        #layer.set_mode(ErrorSimulatorMode.enabled)

        print(image[0].shape)

        batched_inj   = tf.stack([image[0]] * 64)
        golden_label  = tf.stack([mask[0]] * 64)
        
        print(golden_label.shape)
        inj_pred = inj_model.predict(batched_inj)

        inj_pred = create_mask(inj_pred)
        
        iou_0 = calculate_iou(golden_label,inj_pred,0)
        iou_1 = calculate_iou(golden_label,inj_pred,1)
        iou_2 = calculate_iou(golden_label,inj_pred,2)
        mean  = (iou_0 + iou_1 + iou_2) / 3 

        print(f'INJECTION IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}]')

        #pet.display([gold_predict, golden_label[2], inj_pred[2]])



post_fat_segmentation_report(injection_point='conv2d_52')



'''
pred_mask = model.predict(image)
        tot_iou = 0
        for idx in range(0,64):
            #idx = 
            iou_0 = calculate_iou(mask[idx],create_mask(pred_mask,idx),0)
            iou_1 = calculate_iou(mask[idx],create_mask(pred_mask,idx),1)
            iou_2 = calculate_iou(mask[idx],create_mask(pred_mask,idx),2)

            tot_iou += iou_0 + iou_1 + iou_2

            mean  = (iou_0 + iou_1 + iou_2) / 3 
            print(f'IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}]')

        mean_incr = tot_iou / (64*3)

        iou_0 = calculate_iou(mask,create_mask(pred_mask),0)
        iou_1 = calculate_iou(mask,create_mask(pred_mask),1)
        iou_2 = calculate_iou(mask,create_mask(pred_mask),2)
        mean  = (iou_0 + iou_1 + iou_2) / 3 
        print(f'TOT IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}] vs [{mean_incr}]')

'''