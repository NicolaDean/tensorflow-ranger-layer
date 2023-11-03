from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys
import os
import pathlib
from tensorflow.keras.utils import img_to_array, array_to_img
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


F1_score_report = make_dataclass("F1_score_report",[("Layer_name",str),("TH_70_12",float),("TH_70_6",float),("TH_70_3",float),("TH_80_12",float),("TH_80_6",float),("TH_80_3",float),("TH_90_12",float),("TH_90_6",float),("TH_90_3",float)])
    

 #absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
def compute_rgb_similarity(a,b,th=5e-2,r_tol=0.01,a_tol=0.1):

    diff        = np.abs(a - b)
    threshold   = diff <= th
    num_pixel   = np.sum(np.ones_like(a))
    
    return np.sum(threshold) / num_pixel
    
    #Count how many similar pixel in the images
    gold_score = np.sum(np.isclose(a,b,rtol=0.01, atol=0.1))
    #Count num of pixels in general
    num_pixel  = np.sum(np.ones_like(a))
    
    #Percentage of similar pixels
    return gold_score/num_pixel
    
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

def post_fat_segmentation_report(injection_point,version="RGB",out_prefix="test"):

    OUTPUT_NAME    = f"./reports/unet/F1_{out_prefix}.csv"
   

    model = tf.keras.models.load_model("../saved_models/unet_self_drive")

    model.summary()


    x_train,y_train,x_val,y_val = self_drive.loadData(shape=256)
    
    #RAGE TUNE THE YOLO MODEL
    def range_tune(RANGER):
        print("=============FINE TUNING=============")
        RANGER.tune_model_range(x_train, reset=False,verbose=True)

    print("Layers on which we inject faults: ", str(injection_point))
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(model,[injection_point],NUM_INJECTIONS=60,use_classes_ranging=True,range_tuning_fn=range_tune,verbose=True)
    inj_model = RANGER.get_model()
    #yolo_ranger.summary()
    CLASSES.set_model(inj_model)
    CLASSES.disable_all(verbose=False)

    
    #inj_model.summary()
    
    ########################## REPORT #########################

    #Inizialize experiment variables
    print("-------------------------------")
    print(f'Injection on layer {injection_point}')
    print("-------------------------------")



    layer = CLASSES_HELPER.get_layer(inj_model,"classes_" + injection_point,verbose=False)
    assert isinstance(layer, ErrorSimulator)
    layer.set_mode(ErrorSimulatorMode.enabled)

    tot   = 0

    PIXEL_RANGE   = [5e-2,2.5e-2,1e-2]
    THRESHOLDS    = [0.7,0.8,0.9]

    #THRESHOLD_90
    counter = [0,0,0,0,0,0,0,0,0]

    THR_IDX     = 0
    GOOD_SCORE  = 0.7

    progress_bar = tqdm(range(len(x_val)))

    for idx in progress_bar:
        img     = np.expand_dims(x_val[idx],0)
        label   = np.expand_dims(y_val[idx],0)

        golden_pred, actual, mask     = self_drive.predict(img,label, model)
        
        for _ in range(128):
            
            injection_pred , actual, mask = self_drive.predict(img,label, inj_model)          

            c_idx = 0
            #Compute similarity using different configuration of THRESHOLD AND PIXEL RANGE
            for r in PIXEL_RANGE:
                inj_score = compute_rgb_similarity(injection_pred,golden_pred,th=r)
                ig_score  = compute_rgb_similarity(injection_pred,mask,th=r)

                for th in THRESHOLDS:
                    if inj_score < th:
                        counter[c_idx] +=1
                        c_idx += 1

            tot += 1

    for i in range(len(counter)):
        counter[i] = 1 - (counter[i] / tot)

    report = [F1_score_report(injection_point,*counter)]
    
    f1_score_report = pd.DataFrame(report)
    f1_score_report.to_csv(OUTPUT_NAME, mode = 'a', header = False, decimal = ',', sep=';')



import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer"                , action = "store")   
    parser.add_argument("--experiment_name"      , action = "store", default="prefat_rgb")  

    args           = parser.parse_args()
    layer          = args.layer
    prefix         = args.experiment_name

    post_fat_segmentation_report(injection_point=layer,out_prefix=prefix)
