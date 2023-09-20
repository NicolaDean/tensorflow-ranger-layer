import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys

import pathlib
import os

from .custom_loss import *

directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory +  "/../../../keras-yolo3")
from yolo import YOLO, detect_video, compute_iou, compute_F1_score
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from train1 import *

sys.path.append(directory +  "/../../")
from model_helper.run_experiment import *

def build_yolo_classes(WEIGHT_FILE_PATH,classes_path,anchors_path,input_shape,injection_points,classes_enable=True,freeze_body=False,custom_loss=False):

    class_names = get_classes(classes_path)
    anchors     = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    h, w        = input_shape
    
    
    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")

    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) for l in range(3)]
    
    
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_body.load_weights(WEIGHT_FILE_PATH, by_name=True, skip_mismatch=True)
    print('Load weights {}.'.format(WEIGHT_FILE_PATH))


    if freeze_body:
        freeze_body = 2
        # Freeze darknet53 body or freeze all but 3 output layers.
        num = (185, len(model_body.layers)-3)[freeze_body-1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    else:
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True


    vanilla_body = model_body

    if classes_enable:
        RANGER,CLASSES = add_ranger_classes_to_model(model_body,injection_points,NUM_INJECTIONS=30)
        yolo_ranger = RANGER.get_model()
        #yolo_ranger.summary()
        
        CLASSES.set_model(yolo_ranger)
        CLASSES.disable_all()
    else:
        CLASSES = None
        RANGER  = None
        yolo_ranger = vanilla_body
    #model.summary()
    
    if custom_loss:
        golden_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) for l in range(3)]
        loss        = custom_yolo_loss
        loss_input  = [*yolo_ranger.output, *y_true, *golden_true]
        #Define the custom way of combining golden and vanilla yolo_loss
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'custom_loss_combinator': custom_loss_combinator}
    else:
        loss        = yolo_loss
        loss_input = [*yolo_ranger.output, *y_true]
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}

    model_loss = Lambda(loss, 
                        output_shape=(1,), 
                        name        ='yolo_loss',
                        arguments   = arguments
                        )\
                        (loss_input)
    if custom_loss:
        model = Model([yolo_ranger.input, *y_true, *golden_true], model_loss)
    else:
        model = Model([yolo_ranger.input, *y_true], model_loss)

    if not freeze_body:
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    else:
        model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
 
    return model, CLASSES, RANGER, vanilla_body, yolo_ranger