import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys

import pathlib
import os
directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory +  "/../../../keras-yolo3")
from yolo import YOLO, detect_video, compute_iou, compute_F1_score
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

sys.path.append(directory +  "/../../")
from model_helper.run_experiment import *

def build_yolo_classes(classes_path,anchors_path,input_shape):

    class_names = get_classes(classes_path)
    anchors     = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    h, w        = input_shape
    image_input = Input(shape=(None, None, 3))
    
    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")

    '''create the training model'''
    K.clear_session() # get a new session
    
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_body.load_weights('./../../keras-yolo3/yolo_boats_final.h5', by_name=True, skip_mismatch=True)
    print('Load weights {}.'.format('./../../keras-yolo3/yolo_boats_final.h5'))

    for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True

    RANGER,CLASSES = add_ranger_classes_to_model(model_body,"batch_normalization_6",NUM_INJECTIONS=30)
    yolo_ranger = RANGER.get_model()
    yolo_ranger.summary()
    model = yolo_ranger
    
    CLASSES.set_model(model)
    CLASSES.disable_all()

    model.summary()

    model_loss = Lambda(yolo_loss, 
                        output_shape=(1,), 
                        name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}
                        )\
                        ([*model.output, *y_true])
    
    model = Model([model.input, *y_true], model_loss)

    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    return model, CLASSES, RANGER