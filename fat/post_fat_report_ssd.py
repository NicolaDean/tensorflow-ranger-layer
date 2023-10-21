from utils.ssd_model.load_model import load_model_ssd
#run detector on test image
#it takes a little longer on the first run and then runs at normal speed.
import random
import glob
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from utils.metrics.f1_score import recompute_f1

#run detector on test image
#it takes a little longer on the first run and then runs at normal speed.
import random
import glob
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf


from utils.training.gen_golden_annotations import *

sys.path.append("./../../keras-yolo3/")
from yolo import YOLO, detect_video, compute_iou, compute_F1_score

LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

sys.path.append("./")

input_shape = (320,320) # multiple of 32, hw
annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'
annotation_path_valid   = './../../keras-yolo3/valid/_annotations.txt' 
classes_path            = './../../keras-yolo3/train/_classes.txt'         
anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt'


########################## REPORT #########################

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
F1_score_report = make_dataclass("F1_score_report",[("Layer_name",str),("Epoch",int),("Num_wrong_box_shape",int),("Num_wrong_box_count",int),("TOT_Num_Misclassification",int),("Robustness",float),("V_F1_score",float),("I_F1_score",float),("V_accuracy",float),("I_accuracy",float),("V_precision",float),("I_precision",float),("V_recall",float),("I_recall",float),])
    

def post_process_ssd_out(detections,threshold=0.5):
        '''
            ----------------------------------------------------------
            Filter out all boxes with bad score (<50%)
            ----------------------------------------------------------
        '''
        label_id_offset = 1

        scores = detections['detection_scores'][0].numpy()

        detections['detection_classes'][0].numpy() 
        idx = np.argwhere(scores >= threshold)

        scores  = detections['detection_scores'][0].numpy()[idx]
        boxes   = detections['detection_boxes'][0].numpy()[idx]
        classes = np.squeeze((detections['detection_classes'][0].numpy()[idx]).astype(int))

        '''
            ----------------------------------------------------------
            Remove normalization from SSD output boxes
            ----------------------------------------------------------
        '''

        new_boxes = []
        
        for box in boxes:
            
            ymin = box[0][0]  
            xmin = box[0][1]
            ymax = box[0][2]
            xmax = box[0][3]

            (left, right, bottom, top) = (xmin * input_shape[0], xmax * input_shape[0], ymin * input_shape[1], ymax * input_shape[1])
            new_box = [left, bottom, right, top]
            new_boxes.append(new_box)

        classes = classes.tolist() 
        if isinstance(classes, int):
             classes = np.expand_dims(classes, 0).tolist()

        return new_boxes,classes,scores

def post_fat_ssd(model_name='ssd',experiment_name="test",use_classes = True, injection_point='',NUM_INJECTIONS=50,epoch=""):

    OUTPUT_NAME    = f"./reports/{model_name}/{injection_point}/{experiment_name}_epoch_{epoch}.csv"
    OUTPUT_NAME_F1 = f"./reports/{model_name}/{injection_point}/F1_REPORT_{experiment_name}_{injection_point}.csv"

    NUM_ITERATATION_PER_SAMPLE = 50

    import os
  
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(f"./reports/{model_name}/{injection_point}/"):
        
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(f"./reports/{model_name}/{injection_point}/")

    if injection_point=='':
         print("NO INJECTION LAYER SELECTED")
         return
    
    model,CLASSES,RANGER,detect_fn,configs,vanilla_backone,inj_backbone = load_model_ssd(use_classes=use_classes,injection_points=[injection_point])
    
    #IMPORTANT => SET THE proc_img to false to avoid YOLO preprocess on SSD model
    golden_gen_ranger,ranger_size   = get_vanilla_generator('./../../keras-yolo3/train/',32,classes_path,anchors_path,input_shape,random=False, keep_label=True, proc_img=False)
    golden_gen_valid,valid_size     = get_vanilla_generator('./../../keras-yolo3/valid/',1 ,classes_path,anchors_path,input_shape,random=False, keep_label=True, proc_img=False)

    #Range Tune the model
    #RAGE TUNE THE YOLO MODEL
    print("=============FINE TUNING=============")
    for _ in tqdm(range(ranger_size//32)):
        dataset = next(golden_gen_ranger)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)

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
    for sample_id in progress_bar:
        '''
            ----------------------------------------------------------
            Prepare data from dataset
            ----------------------------------------------------------
        '''
        data = next(golden_gen_valid)
        image_data = data[0][0]
        label      = data[2][0]
        label      = label[~np.all(label == 0, axis=1)]

        y_true          = np.hsplit(label,[4,5])
        y_true_boxes    = y_true[0].astype('double')
        y_true_classes  = y_true[1].astype('int')

        y_true_classes = np.reshape(y_true_classes, len(y_true_classes))
        
        #YOLO HA IN FORMATO VOC => xmin xmax ymin ymax
        #y_true_boxes[:, [0, 2]] = y_true_boxes[:, [2, 0]] # ymin xmax xmin ymax
        #y_true_boxes[:, [2, 3]] = y_true_boxes[:, [3, 2]] # ymin xmax ymax xmin
        #y_true_boxes[:, [1, 3]] = y_true_boxes[:, [3, 1]] # ymin xmin ymax xmax

        #YOLO: xmin ymin, xmax ymax

        #y_true_boxes[:, [1, 0]] = y_true_boxes[:, [0, 1]] # ymin xmin xmax,ymax 
        #y_true_boxes[:, [2, 3]] = y_true_boxes[:, [3, 2]] # ymin xmin xmax,ymax
        
        y_true_boxes  = y_true_boxes.tolist()
        y_true_classes= y_true_classes.tolist()

        input_tensor  = tf.convert_to_tensor(image_data, dtype=tf.float32)
        
        '''
            ----------------------------------------------------------
            Predict using SSD model
            ----------------------------------------------------------
        '''
        #Disable Classes
        CLASSES.disable_all(verbose=False)
        #Prediction
        detections, predictions_dict, shapes = detect_fn(input_tensor)
        #PostProcess
        v_out_boxes,v_out_classes,scores = post_process_ssd_out(detections)

        #Compute F1 score
        precision,recall,f1_score, tp, fp, fn = compute_F1_score(y_true_boxes,y_true_classes,v_out_boxes, v_out_classes, iou_th=0.5,verbose=True,convert_format=False)

        print(f'P:{f1_score}')
        
        V_TP += tp
        V_FP += fp
        V_FN += fn
        
        '''
            ----------------------------------------------------------
            Injection Campaign
            ----------------------------------------------------------
        '''
        #Enable Classes
        CLASSES.disable_all(verbose=False)
        #layer = CLASSES_HELPER.get_layer(model._feature_extractor.classification_backbone,"classes_" + injection_point,verbose=False)
        #assert isinstance(layer, ErrorSimulator)
        #layer.set_mode(ErrorSimulatorMode.enabled)

        for _ in range(NUM_INJECTIONS):

            #Prediction
            detections, predictions_dict, shapes = detect_fn(input_tensor)
            #PostProcess
            i_out_boxes,i_out_classes,scores = post_process_ssd_out(detections)
            #Compute partial injection F1 score
            precision,recall,f1_score, tp, fp, fn = compute_F1_score(v_out_boxes,v_out_classes,i_out_boxes, i_out_classes, iou_th=0.5,verbose=False,convert_format=False)

            
            # get injected error id (cardinality, pattern)
            #curr_error_id = layer.error_ids[layer.get_history()[-1]]
            #curr_error_id = np.squeeze(curr_error_id)

            I_TP += tp
            I_FP += fp
            I_FN += fn
                
            #-----------CHECK IF THIS INFERENCE WAS MISCLASSIFIED---------
            if len(v_out_boxes) != len(i_out_boxes):
                num_misclassification_box_shape += 1
            elif fp != 0 or fn != 0:
                num_misclassification_wrong_box += 1
                
            num_misclassification = num_misclassification_box_shape + num_misclassification_wrong_box

            #-------------------------------------------------------------
            report += [Error_ID_report("valid", injection_point, sample_id, curr_error_id[0], curr_error_id[1], 
                                            len(v_out_boxes), len(i_out_boxes), precision, recall, f1_score, tp, fp, fn, None)]

            num_of_injection_comleted += 1

            robustness = 1 - (float(num_misclassification) / float(num_of_injection_comleted))
            progress_bar.set_postfix({'Robu': robustness,'num_exluded': num_excluded,'tot_inj':num_of_injection_comleted})
        
        
    #Stack result of this layer on the report
    report = pd.DataFrame(report)
    report.to_csv(OUTPUT_NAME, mode = 'a', header = False)
    report = []

    #Compute Vanilla   F1 for this layer.
    V_precision,V_recall,V_f1_score,V_accuracy_score = recompute_f1(V_TP,V_FP,V_FN)
    print("Vanilla: Precison: {}, Recall: {}, F1: {}, accuracy: {}".format( V_precision,V_recall,V_f1_score,V_accuracy_score))
    
    #Compute Injection F1 for this layer.
    I_precision,I_recall,I_f1_score,I_accuracy_score = recompute_f1(I_TP,I_FP,I_FN)
    print("Injection: Precison: {}, Recall: {}, F1: {}, accuracy: {}".format( I_precision,I_recall,I_f1_score,I_accuracy_score))

    f1_score_report = [F1_score_report(injection_point,
                                        epoch,
                                        num_misclassification_box_shape,
                                        num_misclassification_wrong_box,
                                        num_misclassification,
                                        robustness,
                                        V_f1_score,I_f1_score,
                                        V_accuracy_score,I_accuracy_score,
                                        V_precision,I_precision,
                                        V_recall,I_recall)]
        
    f1_score_report = pd.DataFrame(f1_score_report)
    f1_score_report.to_csv(OUTPUT_NAME_F1, mode = 'a', header = False)


import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name"      , action = "store", default="ssd")
    parser.add_argument("--experiment_name" , action = "store", default="test")
    parser.add_argument("--layer"           , action = "store", default="")
    parser.add_argument("--prefat",default=False,action = "store_true")


    args            = parser.parse_args()
    model_name      = str(args.model_name)
    experiment_name = str(args.experiment_name)
    layer           = str(args.layer)
    pre_fat         = args.prefat

    post_fat_ssd(model_name=model_name,experiment_name=experiment_name,use_classes=True,injection_point=layer,NUM_INJECTIONS=50,epoch="")



