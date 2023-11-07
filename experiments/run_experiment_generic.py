import tensorflow as tf

import sys
import argparse
import os
from tqdm import tqdm
import pathlib
import numpy as np
LIBRARY_PATH = "/../"

# directory reach
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + LIBRARY_PATH)

from model_helper.run_experiment import *
from utils.dataset import *
from utils.training import *
from utils.injection_campaing import *

import pandas as pd
from dataclasses import make_dataclass

Fault_injection_Report = make_dataclass("Fault_injection_Report", [("model_name",str),("layer_name",str), ("num_of_injection", int), ("misc_nan",int),("misc_clip", int),("misc_rang", int),("nan_robustness",float),("clip_robustness",float),("range_robustness",float)])
Fault_injection_REG_Report = make_dataclass("Fault_injection_REG_Report", [("model_name",str),("layer_name",str), ("num_of_injection", int), ("clip_03", float),("clip_07", float),("clip_1",float),("th_03",float),("th_07",float),("th_1",float)])

Models_statistics = make_dataclass("Models_statistics", [("model_name",str),("dataset",str), ("accuracy",float)])


def classification_head():
    pass

def regression_head():
    pass

def regression_threshold(true,pred,th=0.7):
    diff = np.abs(true-pred)

    #print(f'{true} - {pred} = {diff} => [{(np.sum(diff > th) > 0)}]')

    return (np.sum(diff > th) > 0)


def generate_layer_report(model_name,model,inj_model,DATASET,CLASSES,RANGER,experiment_name,layer_name,NUM_SAMPLE_ITERATION,REGRESSION,USE_RANGER = True):

    (x_train,x_val,y_train,y_val,DATASET_NAME) = DATASET
    print(x_val[0].shape)

    x_val = x_val[:50]
    print(f'INJECTING ON LAYER: [{layer_name}]')
    print(f'NUM SAMPLES = [{x_val.shape[0]}]')
    progress_bar = tqdm(range(x_val.shape[0]))

    if not REGRESSION:
        tot_clip_misc  = 0
        tot_thres_misc = 0
        tot_nan        = 0
        tot_samples    = 0
    else:
        clip_03     = 0
        clip_07     = 0
        clip_1      = 0
        thresh_03 = 0
        thresh_07 = 0
        thresh_1  = 0

    nan_rob     = 0
    clip_rob    = 0
    thresh_rob  = 0
    count       = 0
    tot         = 0

    for idx in progress_bar:

            x = x_val[idx]
            y = y_val[idx]

            x = np.expand_dims(x, 0)
            CLASSES.disable_all(verbose=False)
            pred = model.predict(x, verbose = 0)

            layer = CLASSES_HELPER.get_layer(inj_model,layer_name,verbose=False)
            layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
    
            if REGRESSION:
               
                if not regression_threshold(y,pred,0.7):
                    #print(f"AAAA : {tot}")
                    for idx in range(NUM_SAMPLE_ITERATION):
                        tot += 1
                        
                        RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Layer)
                        inj_pred = inj_model.predict(x,verbose=False)

                        clip_03 += regression_threshold(inj_pred,pred,0.3)
                        clip_07 += regression_threshold(inj_pred,pred,0.7)
                        clip_1  += regression_threshold(inj_pred,pred,1)

                        RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Layer)   
                        inj_pred = inj_model.predict(x,verbose=False)

                        thresh_03 += regression_threshold(inj_pred,pred,0.3)
                        thresh_07 += regression_threshold(inj_pred,pred,0.7)
                        thresh_1  += regression_threshold(inj_pred,pred,1)

                        #print("--------------------")
                        progress_bar.set_postfix({'tot':tot,'clip 0.3': clip_03,'clip 07': clip_07,'th 0.3': thresh_03,'th 07': thresh_07})
                        
            else:
                #IF NOT MISCLASSIFIED
                if np.argmax(pred) == y:
                    tot_samples += NUM_SAMPLE_ITERATION
                    #Create a batch
                    BATCH_SIZE = 64
                    #Generate an array of shape (batch,IMG) where each image is the same as current sample
                    x_batch,y_batch = gen_batch(x[0],y,batch_size=NUM_SAMPLE_ITERATION)
                    
                    y_batch = tf.cast(y_batch,tf.int32)
                    #print(f'AAAAAAAAAAAAAAA:{x_batch.shape}')
                    RANGER.set_ranger_mode(RangerModes.RangeTuning)
                    pred        = inj_model.predict(x_batch,batch_size=BATCH_SIZE,verbose=False)
                    pred        = tf.cast(tf.argmax(pred, 1),tf.int32)
                    nan_misc   = tf.reduce_sum(tf.cast(tf.not_equal(y_batch,pred), tf.int32)).numpy()
                    tot_nan    += nan_misc
                    nan_acc = tot_nan / tot_samples
                    nan_rob = 1-nan_acc
                    clip_rob    = 0
                    thresh_rob  = 0

                    
                    RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Layer)
                    pred        = inj_model.predict(x_batch,batch_size=BATCH_SIZE,verbose=False)
                    pred        = tf.cast(tf.argmax(pred, 1),tf.int32)
                    clip_misc   = tf.reduce_sum(tf.cast(tf.not_equal(y_batch,pred), tf.int32)).numpy()
                    tot_clip_misc += clip_misc
                    clip_acc    = tot_clip_misc / tot_samples
                    clip_rob    = 1 - clip_acc
                    #                print(f'PRED SHAPE: [{pred.shape}] vs [{y_batch.shape}]')
                    RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Ranger,RangerGranularity.Layer)   
                    pred        = inj_model.predict(x_batch,batch_size=BATCH_SIZE,verbose=False)
                    pred        = tf.cast(tf.argmax(pred, 1),tf.int32)
                    thres_misc  = tf.reduce_sum(tf.cast(tf.not_equal(y_batch,pred), tf.int32)).numpy()
                    tot_thres_misc += thres_misc
                    thresh_acc  = tot_thres_misc / tot_samples
                    thresh_rob  = 1 - thresh_acc

                    progress_bar.set_postfix({'Clip_rob': clip_rob,'Thresh_rob': thresh_rob,'tot_samples': tot_samples})
                    
    if not REGRESSION:
        line_report = Fault_injection_Report(model_name,layer_name,tot_samples,tot_nan,tot_clip_misc,tot_thres_misc,nan_rob,clip_rob,thresh_rob)
    else:
        rob_clip03 = 1- clip_03/tot
        rob_clip07 = 1- clip_07/tot
        rob_clip1  = 1- clip_1/tot

        rob_th03 = 1- thresh_03/tot
        rob_th07 = 1- thresh_07/tot
        rob_th1  = 1- thresh_1/tot

        line_report = Fault_injection_REG_Report(model_name,layer_name,tot,rob_clip03,rob_clip07,rob_clip1,rob_th03,rob_th07,rob_th1)
    return line_report


def generate_report(model_name,model,DATASET,experiment_name,NUM_SAMPLE_ITERATION,RESUME_LAYER,START_IDX,REGRESSION):

    use_layer = True
    if RESUME_LAYER != "NONE":
        print(f"RESUME FROM LAYER: [{RESUME_LAYER}]")
    use_layer = False

    layer_names = []
    counter     = 0
    start_cnt   = 0

    if len(model.layers) < START_IDX:
        print(f"\033[0;31mEXPERIMENT END FOR {model_name}\033[0m")
        exit()
    
    for layer in model.layers:
        
        if layer.name == RESUME_LAYER or start_cnt >= START_IDX:
            use_layer = True

        if use_layer:
            counter += 1
            if counter <= 5:
                print(f'USE LAYER: [{layer.name}]')
                layer_names.append(layer.name)
            else:
                continue
        else:
            print(f'SKIPPED LAYER: [{layer.name}]')
        
        start_cnt +=1

    print(f"LEN: {len(layer_names)}")  
    print(layer_names)   

    def range_tune(RANGER):
        #Range Tuning
        #TUNE THE LAYERS RANGE DOMAIN
        print("==============RANGE TUNING================")
        RANGER.tune_model_range(x_train[:200],verbose=True)#DECOMMENT

    RANGER,CLASSES = add_ranger_classes_to_model(model,layer_names,NUM_INJECTIONS=50,use_classes_ranging=True,range_tuning_fn=range_tune)
    inj_model = RANGER.get_model()
    #yolo_ranger.summary()
    CLASSES.set_model(inj_model)
    CLASSES.disable_all(verbose=False)

    #inj_model.summary()

   

    layer_names = CLASSES.injection_points


    report = []
    #Report
    for layer in layer_names:
        line = generate_layer_report(model_name,model,inj_model,DATASET,CLASSES,RANGER,experiment_name,layer,NUM_SAMPLE_ITERATION,REGRESSION)
        report = [line]
        
        report = pd.DataFrame(report)
        report.to_csv(f"./classification_report/{model_name}_{DATASET[4]}_summary.csv",header=False,mode = 'a', decimal = ',', sep=';')
        exit()
    line_report = Fault_injection_Report("END","END",0,0,0,0,0)
    report = pd.DataFrame([line_report])
    report.to_csv(f"./classification_report/{model_name}_{DATASET[4]}_summary.csv",header=False,mode = 'a', decimal = ',', sep=';')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name" , default="generic_experiment", action = "store")
    parser.add_argument("--model"           , default="vgg16", action = "store")
    parser.add_argument("--dataset"         , default="MNIST", action = "store")
    parser.add_argument("--no_train"        , default=False , action = "store_true")
    parser.add_argument("--no_report"       , default=False,  action = "store_true")
    parser.add_argument("--epochs"          , default=5,action="store")
    parser.add_argument("--regression"      , default=False,action="store_true")
    parser.add_argument("--gen_model_statistics",default=False,action="store_true")
    parser.add_argument("--start_at",default=0,action="store")
    parser.add_argument("--resume_from",default="NONE",action="store") #Riprende dal layer selezionato
    parser.add_argument("--input_shape",default=32,action="store") 

    args            = parser.parse_args()

    MODEL           = str(args.model)
    EXP_NAME        = str(args.experiment_name)
    DATASET_NAME    = str(args.dataset)
    DO_TRAIN        = not bool(args.no_train)
    DO_REPORT       = not bool(args.no_report)
    DO_STATISTICS   = bool(args.gen_model_statistics)
    EPOCHS          = int(args.epochs)
    RESUME_LAYER    = str(args.resume_from)
    START_AT        = int(args.start_at)
    REGRESSION      = bool(args.regression)
    INPUT_SHAPE     = int(args.input_shape)
    INPUT_SHAPE     = (INPUT_SHAPE,INPUT_SHAPE)


    #Load chosen dataset
    x_train,x_val,y_train,y_val,NUM_CLASSES = load_dataset(DATASET=DATASET_NAME,shape=INPUT_SHAPE)

    #Extract numclasses and input shape from dataset
    INPUT_SHAPE = x_train[0].shape

    print(f'INPUT SHAPE: {INPUT_SHAPE}')

    #Load model and preprocess function
    model,preprocess_fn = load_model(MODEL=MODEL,NUM_CLASSES=NUM_CLASSES,INPUT_SHAPE=INPUT_SHAPE,REGRESSION=REGRESSION)
    
    #Preprocess data for the specific model
    if REGRESSION:
        x_train /= 255
        x_val /= 255
    else:
        x_train = preprocess_fn(x_train)
        x_val   = preprocess_fn(x_val)

    DATASET     = (x_train,x_val,y_train,y_val,DATASET_NAME)
    
    MODEL_PATH = f"./saved_models/{MODEL}_{DATASET_NAME}"

    if DO_TRAIN:
        train_model(model,DATASET,NUM_CLASSES,EPOCHS,REGRESSION)
        model.save(MODEL_PATH)
    else:
        print("Load Model from path")
        model       = tf.keras.models.load_model(MODEL_PATH)
    
    if DO_STATISTICS:
        if not REGRESSION:
            y_val_cat   = keras.utils.to_categorical(y_val, NUM_CLASSES)
        else:
            y_val_cat = y_val
            
        loss,acc    = model.evaluate(x_val,y_val_cat,)
        line        = Models_statistics(MODEL,DATASET_NAME,acc)
        statistics  = pd.DataFrame([line])
        statistics.to_csv("./classification_report/_models_statistics.csv",mode = 'a', header = False, decimal = ',', sep=';')
        exit()

    if DO_REPORT:
        generate_report(MODEL,model,DATASET,experiment_name=EXP_NAME,NUM_SAMPLE_ITERATION=128,RESUME_LAYER=RESUME_LAYER,START_IDX=START_AT,REGRESSION=REGRESSION)


