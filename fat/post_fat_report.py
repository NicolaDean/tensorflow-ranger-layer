import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys
import argparse
import os
from tqdm import tqdm

sys.path.append("./../../keras-yolo3/")

from yolo import YOLO, detect_video, compute_iou, compute_F1_score
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from train import *


'''
-----------------------------
-------DESIRED OUTPUT:-------
-----------------------------
- F1_injection the F1 with injection always ON  (for best epoch)
- F1_vanilla   the F1 with injection always OFF (for best epoch)
- Best_Epoch   the epoch with the highest F1_inj mantaining a F1_van higher than threshold (no peggioramenti)
- Misc_E[i]    Num of misclassification with checkpoint at epoch [i]
Layer -- F1_injection -- F1_vanilla -- Best Epoch-- Misc E5 -- Misc E10 -- Misc E15 -- Misc E20 -- Misc E25


'''
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

sys.path.append("./")

def recompute_f1(TP,FP,FN):
        if (TP + FP) != 0:
            precision = TP / (TP + FP) 
        else:
            precision = 0

        if (TP + FN) != 0:
            recall    = TP / (TP + FN)
        else:
            recall = 0

        if (precision + recall) != 0:
            f1_score  = (2*precision*recall)/(precision + recall)
        else:
            f1_score = None

        if (FP+FN+TP) != 0:
            accuracy_score = (TP) / (FP+FN+TP)
        else:
            accuracy_score = None

        return precision,recall,f1_score,accuracy_score


def generate_report(check_points_path,selected_layer,epoch=5,out_prefix="yolo_boats_POST_FAT"):

    OUTPUT_NAME    = f"./reports/{selected_layer[0]}/{out_prefix}_epoch_{epoch}.csv"
    OUTPUT_NAME_F1 = f"./reports/{selected_layer[0]}/F1_REPORT_{out_prefix}_{selected_layer}.csv"

    NUM_ITERATATION_PER_SAMPLE = 50

    # './export/_annotations.txt'
    annotation_path_train = './../../keras-yolo3/train/_annotations.txt'
    annotation_path_test  = './../../keras-yolo3/test/_annotations.txt'
    annotation_path_valid  = './../../keras-yolo3/valid/_annotations.txt'
    #log_dir = 'logs/000/'
    # './export/_annotations.txt'
    classes_path = './../../keras-yolo3/train/_classes.txt'
    anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    with open(annotation_path_train) as f:
        train_lines = f.readlines()
    
    with open(annotation_path_test) as f:
        test_lines = f.readlines()
    
    with open(annotation_path_valid) as f:
        valid_lines = f.readlines()

    print(f"Found {len(train_lines)} elements in {annotation_path_train}")
    print(f"Found {len(test_lines)} elements in {annotation_path_test}")
    print(f"Found {len(valid_lines)} elements in {annotation_path_valid}")


    train_gen = data_generator_wrapper('./../../keras-yolo3/train/',train_lines, 32, input_shape, anchors, num_classes, random = False)
    test_gen  = data_generator_wrapper('./../../keras-yolo3/test/',test_lines, 1, input_shape, anchors, num_classes, random = False)
    valid_gen = data_generator_wrapper('./../../keras-yolo3/valid/',valid_lines, 1, input_shape, anchors, num_classes, random = False)

    #weights = './../../keras-yolo3/yolo_boats_final.h5'

    class args:
        def __init__ (self, model_path = check_points_path,
                            anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt',
                            classes_path =  './../../keras-yolo3/train/_classes.txt',
                            score = 0.3,
                            iou = 0.45,
                            model_image_size = (416, 416),
                            gpu_num = 1):
            self.model_path  = model_path
            self.anchors_path  = anchors_path
            self.classes_path  = classes_path
            self.score  = score
            self.iou  = iou
            self.model_image_size  = model_image_size
            self.gpu_num  = gpu_num

    argss = args()
    print("loading yolo")
    yolo        = YOLO(**vars(argss))
    yolo_faulty = YOLO(**vars(argss))
    yolo_model = yolo.yolo_model

    layer_names = selected_layer

    print("Layers on which we inject faults: ", str(layer_names))
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(yolo.yolo_model,layer_names,NUM_INJECTIONS=60)
    yolo_ranger = RANGER.get_model()
    #yolo_ranger.summary()
    CLASSES.set_model(yolo_ranger)
    CLASSES.disable_all(verbose=False)

    yolo_faulty = yolo_ranger
   
    #RAGE TUNE THE YOLO MODEL
    print("=============FINE TUNING=============")
    for _ in tqdm(range(12)):
        dataset = next(train_gen)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)

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
    
    #report = pd.DataFrame(columns = Error_ID_report.__annotations__.keys())
    #report.to_csv("../reports/yolo_boats_test_NOrandom.csv")
    report = []
    
    from PIL import Image
    
    # VALIDATION SET

    #for all layers we want to inject faults
    for layer_name in layer_names:
        print("-------------------------------")
        print(f'Injection on layer {layer_name}')
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

        #for all test samples
        progress_bar = tqdm(range(len(valid_lines)))
        for sample_id in progress_bar:
            EMPTY_BOX_FLAG = False

            #load data
            dataset = next(valid_gen)
            data   = dataset[0][0]
            label  = dataset[2][0]
            label  = label[~np.all(label == 0, axis=1)]
            
            y_true          = np.hsplit(label,[4,5])
            y_true_boxes    = y_true[0].astype('double')
            y_true_classes  = y_true[1].astype('int')

            y_true_classes = np.reshape(y_true_classes, len(y_true_classes))

            y_true_boxes[:, [0, 1]] = y_true_boxes[:, [1, 0]]
            y_true_boxes[:, [2, 3]] = y_true_boxes[:, [3, 2]]

            y_true_boxes  = y_true_boxes.tolist()
            y_true_classes= y_true_classes.tolist()

            img = np.uint8(data[0]*255)
            img = Image.fromarray(img)
            f1 = img
            f2 = copy.deepcopy(img)

            #vanilla prediction
            yolo.yolo_model = yolo_model
            r_image,v_out_boxes, v_out_scores, v_out_classes = yolo.detect_image(f1,y_true=False,verbose=False)
            r_image         = np.asarray(r_image)

            '''
            if v_out_boxes.shape[0] == 0:
                
                report += [Error_ID_report("valid", layer_name, sample_id, np.nan, np.nan, 
                                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                           "Golden prediction has no boxes")]
                
                num_empty += 1
                #continue
            '''

            #150 faults injections are performed
            yolo.yolo_model = yolo_faulty
            CLASSES.disable_all(verbose=False)
            layer = CLASSES_HELPER.get_layer(yolo_faulty,"classes_" + layer_name,verbose=False)
            assert isinstance(layer, ErrorSimulator)
            layer.set_mode(ErrorSimulatorMode.enabled)


            for _ in range(NUM_ITERATATION_PER_SAMPLE):
                err = ""
                #Compute Inference with injection
                try:
                    r_image_faulty,out_boxes, out_scores, out_classes  = yolo.detect_image(f2,y_true=False, no_draw = True,verbose=False)
                    r_image_faulty  = np.asarray(r_image_faulty)
                except Exception as e:
                    err = e

                # get injected error id (cardinality, pattern)
                curr_error_id = layer.error_ids[layer.get_history()[-1]]
                curr_error_id = np.squeeze(curr_error_id)

                #Exclude sample if some error occurred
                if err != "":
                    report += [Error_ID_report("valid", layer_name, sample_id, curr_error_id[0], curr_error_id[1], 
                                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, err)]
                    num_excluded += 1
                    continue
            

                #Avoid crashing when IOU is not computabel due to "division by zero"
                '''
                if not EMPTY_BOX_FLAG:
                    try:
                        iou_curr = compute_iou(v_out_boxes,v_out_classes,out_boxes, out_scores, out_classes).numpy()[0]
                    except:
                        iou_curr = np.nan
                else:
                    iou_curr = np.nan
                '''

                #Compute partial injection F1 score
                precision,recall,f1_score, tp, fp, fn = compute_F1_score(v_out_boxes,v_out_classes,out_boxes, out_classes, iou_th=0.5,verbose=False)

                I_TP += tp
                I_FP += fp
                I_FN += fn
                
                #-----------CHECK IF THIS INFERENCE WAS MISCLASSIFIED---------
                if len(v_out_boxes.shape) != len(out_boxes.shape):
                    num_misclassification_box_shape += 1
                elif fp != 0 or fn != 0:
                    num_misclassification_wrong_box += 1
                
                num_misclassification = num_misclassification_box_shape + num_misclassification_wrong_box

                #-------------------------------------------------------------
                report += [Error_ID_report("valid", layer_name, sample_id, curr_error_id[0], curr_error_id[1], 
                                            v_out_boxes.shape[0], out_boxes.shape[0], precision, recall, f1_score, tp, fp, fn, err)]
                    
                #-----------VANILLA F1 SCORE----------------------------------

                #Compute partial vanilla F1 score
                #CHECK CONTROLLARE SE L'ORDINE DI TRUE_LAB / V_BOX è IMPORTANTE AI FINI DI FP e FN
                precision,recall,f1_score, tp, fp, fn = compute_F1_score(y_true_boxes,y_true_classes,v_out_boxes, v_out_classes, iou_th=0.5,verbose=False)

                V_TP += tp
                V_FP += fp
                V_FN += fn

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

        f1_score_report = [F1_score_report(layer_name,
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

 ###################### END REPORT ##########################


CHECKPOINT_PATH = "./results/FREQUENCY_0.5__SINGLE_LAYER_batch_normalization_5/FREQUENCY_0.5__SINGLE_LAYER_batch_normalization_5-ep030.h5"
SELECTED_LAYERS = ["batch_normalization_5"]

#CHECKPOINT_PATH = "./results/../../../keras-yolo3/yolo_boats_final.h5"

if __name__ == '__main__':
    #generate_report(CHECKPOINT_PATH,SELECTED_LAYERS,epoch=200,out_prefix="PRE_FAT")

    #exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action = "store")
    parser.add_argument("--epoch", action = "store")
    parser.add_argument("--experiment_name",action = "store")
    parser.add_argument("--layer",action = "store")
    parser.add_argument("--prefat",default=False,action = "store_true")


    args            = parser.parse_args()
    prefix          = args.checkpoint
    epoch           = str(args.epoch)
    experiment_name = str(args.experiment_name)
    layer           = str(args.layer)
    pre_fat         = args.prefat

    if pre_fat:
        CHECKPOINT_PATH = "./results/../../../keras-yolo3/yolo_boats_final.h5"
        SELECTED_LAYERS = [layer]
        generate_report(CHECKPOINT_PATH,SELECTED_LAYERS,epoch=epoch,out_prefix=experiment_name)
        exit()
        
    SELECTED_LAYERS = [layer]

    while len(epoch) < 3:
        epoch = "0"+epoch

    for file_name in os.listdir("results/"+prefix):
        if "-ep"+epoch in file_name:
            CHECKPOINT_PATH =  "./results/"+ prefix + "/" + file_name
            break
    
    generate_report(CHECKPOINT_PATH,SELECTED_LAYERS,epoch=epoch,out_prefix=experiment_name)