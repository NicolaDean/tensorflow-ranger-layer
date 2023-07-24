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

class Obj_metrics_callback(keras.callbacks.Callback):
    
    def __init__(self,model_body,valid_path,classes_path,anchors_path,input_shape,frequency=5):
        super().__init__()
        self.valid_gen,self.valid_size = get_vanilla_generator(valid_path,1,classes_path,anchors_path,input_shape,random=True,keep_label=True)

        class_names     = get_classes(classes_path)
        self.anchors     = get_anchors(anchors_path)
        self.num_classes = len(class_names)
        self.input_shape = input_shape
        self.model_body  = model_body

        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.frequency):
            return
        
        print("\n")
        print("-----------------------")
        print("-----F1 SCOREEE--------")
        print("-----------------------")

        TP,FP,FN = 0,0,0
        for idx in tqdm(range(0,self.valid_size)):
            data = next(self.valid_gen)
            image_data = data[0][0]
            label      = data[2][0]
            label      = label[~np.all(label == 0, axis=1)]

            y_true          = np.hsplit(label,[4,5])
            y_true_boxes    = y_true[0].astype('double')
            y_true_classes  = y_true[1].astype('int')

            y_true_classes = np.reshape(y_true_classes, len(y_true_classes))

            y_true_boxes  = y_true_boxes.tolist()
            y_true_classes= y_true_classes.tolist()

            yolo_out = self.model_body.predict(image_data,verbose=False)
            
            boxes, scores, classes = yolo_eval(yolo_out, self.anchors,self.num_classes, self.input_shape,score_threshold=0.7, iou_threshold=0.5)

            precision,recall,f1, tp, fp, fn = compute_F1_score(y_true_boxes,y_true_classes,boxes, classes, iou_th=0.5,verbose=False)

            TP += tp
            FP += fp
            FN += fn
            #Compute Precision, Recall, F1_score
        if TP + FP != 0:
            precision = TP / (TP + FP) 
        else:
            precision = 0

        if TP + FN != 0:
            recall    = TP / (TP + FN)
        else:
            recall = 0

        if precision + recall != 0:
            f1_score  = (2*precision*recall)/(precision + recall)
        else:
            f1_score = None

        keys = list(logs.keys())
        print("End epoch {} of training; Precison: {}, Recall: {}, F1: {}".format(epoch, precision,recall,f1_score))