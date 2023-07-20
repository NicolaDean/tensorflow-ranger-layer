import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys

sys.path.append("./../../keras-yolo3/")

from yolo import YOLO, detect_video, compute_iou, compute_F1_score
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from train import *

LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

sys.path.append("./")


def _main():

    input_shape = (416,416) # multiple of 32, hw
    annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'
    annotation_path_valid   = './../../keras-yolo3/valid/_annotations.txt' 
    classes_path            = './../../keras-yolo3/train/_classes.txt'         
    anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt'

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

    #model = create_model(input_shape, anchors, num_classes,
    #        freeze_body=2, weights_path='./../../keras-yolo3/yolo_boats_final.h5') # make sure you know what you freeze
    
    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    
    with open(annotation_path_train) as f:
        train_lines = f.readlines()
    
    with open(annotation_path_valid) as f:
        valid_lines = f.readlines()

    np.random.seed(10101)
    np.random.seed(None)
    num_val = len(valid_lines)
    num_train = len(train_lines)

    train_gen = data_generator_wrapper('./../../keras-yolo3/train/',train_lines, 32, input_shape, anchors, num_classes, random = False)
    
    
    #RANGE TUNE THE YOLO MODEL
    print("=============RANGE TUNING=============")
    for _ in range(12):
        dataset = next(train_gen)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)
    
    #UNLOCK CLASSES
    layer = CLASSES_HELPER.get_layer(model,"classes_batch_normalization_6")
    assert isinstance(layer, ErrorSimulator)
    layer.set_mode(ErrorSimulatorMode.enabled)

    model_loss = Lambda(yolo_loss, 
                        output_shape=(1,), 
                        name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}
                        )\
                        ([*model.output, *y_true])
    
    model = Model([model.input, *y_true], model_loss)

    train_gen = data_generator_wrapper('./../../keras-yolo3/train/',train_lines, 32, input_shape, anchors, num_classes, random = True)
    valid_gen = data_generator_wrapper('./../../keras-yolo3/valid/',valid_lines, 32, input_shape, anchors, num_classes, random = True)


    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    batch_size = 32 # note that more GPU memory is required after unfreezing the body
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit(train_gen,
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=valid_gen,
        validation_steps=max(1, num_val//batch_size),
        epochs=100)
    model.save_weights('trained_weights_final.h5')

    # Further training if needed.



if __name__ == '__main__':
    _main()
