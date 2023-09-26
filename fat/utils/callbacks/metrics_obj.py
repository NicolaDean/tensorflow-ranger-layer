#Create a callback to print IOU, MAP, F1 metric realtime during training of model

import tensorflow as tf
import keras
import numpy as np
import sys
import pathlib
from tqdm import tqdm

from ..training.gen_golden_annotations import *

directory = str(pathlib.Path(__file__).parent.parent.absolute()) +  "/../../../keras-yolo3"
sys.path.append(directory)

from yolo3.model import yolo_eval
from yolo import compute_F1_score, compute_iou

directory = str(pathlib.Path(__file__).parent.parent.absolute()) +  "/../training/gen"
sys.path.append(directory)

def compute_validation_f1(model_body,valid_gen, valid_size,anchors,num_classes, input_shape):
        TP,FP,FN,F1 = 0,0,0,0
        for idx in tqdm(range(0,valid_size)):
            data = next(valid_gen)
            image_data = data[0][0]
            label      = data[2][0]
            label      = label[~np.all(label == 0, axis=1)]

            y_true          = np.hsplit(label,[4,5])
            y_true_boxes    = y_true[0].astype('double')
            y_true_classes  = y_true[1].astype('int')

            y_true_classes = np.reshape(y_true_classes, len(y_true_classes))

            y_true_boxes[:, [0, 1]] = y_true_boxes[:, [1, 0]]
            y_true_boxes[:, [2, 3]] = y_true_boxes[:, [3, 2]]

            y_true_boxes  = y_true_boxes.tolist()
            y_true_classes= y_true_classes.tolist()

            yolo_out = model_body.predict(image_data,verbose=False)
            
            boxes, scores, classes = yolo_eval(yolo_out, anchors,num_classes, input_shape,score_threshold=0.3, iou_threshold=0.45)
            '''
            print("TRUE")
            print(y_true_boxes)
            print(y_true_classes)
            print("PREDICTED")
            print(boxes)
            print(classes)
            '''

            precision,recall,f1, tp, fp, fn = compute_F1_score(y_true_boxes,y_true_classes,boxes, classes, iou_th=0.5,verbose=False)
            F1 += f1
            TP += tp
            FP += fp
            FN += fn
            #Compute Precision, Recall, F1_score
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
            Accuracy = (TP) / (FP+FN+TP)
        else:
            Accuracy = None

        F1 = F1 / valid_size
        
        return precision,recall,f1_score,Accuracy

class Obj_metrics_callback(keras.callbacks.Callback):
    
    def __init__(self,model_body,valid_path,classes_path,anchors_path,input_shape,frequency=5, CLASSES = None, mixed_callback = None):
        super().__init__()
        self.valid_gen,self.valid_size = get_vanilla_generator(valid_path,1,classes_path,anchors_path,input_shape,random=False,keep_label=True)

        class_names      = get_classes(classes_path)
        self.anchors     = get_anchors(anchors_path)
        self.num_classes = len(class_names)
        self.input_shape = input_shape
        self.model_body  = model_body

        self.frequency = frequency
        self.CLASSES = CLASSES
        self.mixed_callback = mixed_callback

    def on_epoch_end(self, epoch, logs=None):
        epoch +=1
        if (epoch % self.frequency) == 0:
            print("\n")
            print("-----------------------")
            print("-----F1 SCORE--------")
            print("-----------------------")
            precision,recall,f1_score,Accuracy = compute_validation_f1(self.model_body,self.valid_gen,self.valid_size,self.anchors,self.num_classes, self.input_shape)
            keys = list(logs.keys())
            print("End epoch {} of training; Precison: {}, Recall: {}, F1: {}, accuracy: {}".format(epoch, precision,recall,f1_score,Accuracy))

        if self.mixed_callback != None:
            if epoch%(self.mixed_callback.num_epochs_switch*2) == 0 and self.mixed_callback.v3:
                print("CURRENT F1 SCORE COMPUTATION - NO INJECTIONS")
                self.CLASSES.disable_all()
                precision,recall,f1_score,Accuracy = compute_validation_f1(self.model_body,self.valid_gen,self.valid_size,self.anchors,self.num_classes, self.input_shape)
                print("f1 target = {}    f1 current = {}".format(self.mixed_callback.f1_target, f1_score))
                self.mixed_callback.f1_current = f1_score
                print("f1 target = {}    f1 current = {}".format(self.mixed_callback.f1_target, f1_score))


    def on_train_end(self, logs=None):
        self.on_epoch_end(1,logs)