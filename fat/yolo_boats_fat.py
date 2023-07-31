import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys

from utils.training.gen_golden_annotations import get_golden_generator

sys.path.append("./../../keras-yolo3/")

from yolo import YOLO, detect_video, compute_iou, compute_F1_score
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_eval
from yolo3.utils import get_random_data, letterbox_image

from train1 import *

LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

sys.path.append("./")


def _main():
    annotation_path_train = './../../keras-yolo3/train/_annotations.txt'
    annotation_path_valid  = './../../keras-yolo3/valid/_annotations.txt' 
    classes_path = './../../keras-yolo3/train/_classes.txt'         
    anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_body.load_weights('./../../keras-yolo3/yolo_boats_final.h5', by_name=True, skip_mismatch=True)
    print('Load weights {}.'.format('./../../keras-yolo3/yolo_boats_final.h5'))
    
    RANGER,CLASSES = add_ranger_classes_to_model(model_body,["conv2d"],NUM_INJECTIONS=30)
    yolo_ranger = RANGER.get_model()
    yolo_ranger.summary()
    model_body = yolo_ranger
    
    CLASSES.set_model(model_body)
    CLASSES.disable_all()
    
    #model = create_model(input_shape, anchors, num_classes,
    #        freeze_body=2, weights_path='./../../keras-yolo3/yolo_boats_final.h5') # make sure you know what you freeze
    
    

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
    print("=============FINE TUNING=============")
    for _ in range(1):
        dataset = next(train_gen)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)
    
    '''
    
    #CREATION ANNOTATIONS FROM GOLDEN MODEL PREDICTIONS
    train_gen = data_generator_wrapper('./../../keras-yolo3/train/',train_lines, 1, input_shape, anchors, num_classes, random = False)
    valid_gen = data_generator_wrapper('./../../keras-yolo3/valid/',valid_lines, 1, input_shape, anchors, num_classes, random = False)

    from PIL import Image
    
    golden_train_lines = []
    #Create Golden Annotations for Training
    for i in range(371):
        name_file = train_lines[i].split()[0]
        dataset = next(train_gen)
        image  = dataset[0][0][0]
        img = np.uint8(image*255)
        image = Image.fromarray(img)
        new_image_size = (image.width - (image.width % 32),
                                image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        model_output = model_body.predict(image_data)
        #annotations = (boxes, scores, classes)
        annotations = yolo_eval(model_output, anchors, num_classes, input_shape, score_threshold = 0.7, iou_threshold = 0.5)
        
        out_boxes, out_scores, out_classes = annotations

        assert annotations[0].shape[1] == 4
        assert len(annotations[2].shape) >= 1

        y_golden = np.column_stack((out_boxes,out_classes))
        y_golden[:, [0, 1]] = y_golden[:, [1, 0]]
        y_golden[:, [2, 3]] = y_golden[:, [3, 2]]
        print(f'Golden labels {y_golden}')

        resultString = name_file
        for a in y_golden: 
            resultString += " "+','.join(map(str, a.astype(int)))
        print(resultString)
        golden_train_lines.append(resultString)
    '''

    '''
    #Create Golden annotation for Validation
    golden_valid_lines = []

    for i in range(105):
        name_file = valid_lines[i].split()[0]
        dataset = next(valid_gen)
        data   = dataset[0][0]
        image_data = data
        model_output = model_body.predict(image_data)
        #annotations = (boxes, scores, classes)
        annotations = yolo_eval(model_output, anchors, num_classes, input_shape, score_threshold = 0.5, iou_threshold = 0.5)
        
        assert annotations[0].shape[1] == 4
        assert len(annotations[2].shape) >= 1
        
        resultString = name_file
        for i, a in enumerate(annotations[0]): 
            resultString += " "+','.join(map(str, a.numpy().astype(int)))
            resultString += ","+str(annotations[2][i].numpy())
        print(resultString)
        golden_valid_lines.append(resultString)    
    '''
    '''
    #UNLOCK CLASSES
    layer = CLASSES_HELPER.get_layer(model_body,"classes_conv2d")
    assert isinstance(layer, ErrorSimulator)
    layer.set_mode(ErrorSimulatorMode.enabled)
    
    '''



    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    
    #model = RANGER.get_model()
    RANGER.set_ranger_mode(RangerModes.Disabled)

    model = Model([model_body.input, *y_true], model_loss)
    model.summary()

    for i in range(len(model.layers)):
            model.layers[i].trainable = True

    #TRAINING WITH CLASSES INJECTING FAULTS AT EACH SAMPLE
    
    batch_size = 8 # note that more GPU memory is required after unfreezing the body
    '''
    train_gen = data_generator_wrapper('./../../keras-yolo3/train/',golden_train_lines, batch_size, input_shape, anchors, num_classes, random = False)
    valid_gen = data_generator_wrapper('./../../keras-yolo3/valid/',golden_valid_lines, batch_size, input_shape, anchors, num_classes, random = False)
    '''

    train_gen, size_train = get_golden_generator(model_body, './../../keras-yolo3/train/', batch_size, classes_path, anchors_path, input_shape, random = False)
    #train_gen = data_generator_wrapper('./../../keras-yolo3/train/',train_lines, batch_size, input_shape, anchors, num_classes, random = True)
    valid_gen, size_valid = get_golden_generator(model_body, './../../keras-yolo3/valid/', batch_size, classes_path, anchors_path, input_shape, random = False)

    #valid_gen = data_generator_wrapper('./../../keras-yolo3/valid/',valid_lines, batch_size, input_shape, anchors, num_classes, random = True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    history = model.fit(train_gen,
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=valid_gen,
        validation_steps=max(1, num_val//batch_size),
        epochs=10, 
        callbacks = [reduce_lr])

    


if __name__ == '__main__':
    _main()
