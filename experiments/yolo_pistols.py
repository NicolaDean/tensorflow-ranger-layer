import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys

sys.path.append("./../../keras-yolo3/")

from yolo import YOLO, detect_video
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from train import *

LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

sys.path.append("./")

def _main():
    # './export/_annotations.txt'
    annotation_path = './../../keras-yolo3/pistols_dataset/export/_annotations.txt'
    #log_dir = 'logs/000/'
    # './export/_annotations.txt'
    classes_path = './../../keras-yolo3/pistols_dataset/export/_classes.txt'
    anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    with open(annotation_path) as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} elements in {annotation_path}")



    gen = data_generator_wrapper('./../../keras-yolo3/pistols_dataset/export/',lines, 1, input_shape, anchors, num_classes)

    

    class args:
        def __init__ (self, model_path = './../../keras-yolo3/pistols.h5',
                            anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt',
                            classes_path =  './../../keras-yolo3/pistols_dataset/export/_classes.txt',
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

    yolo_model.summary()

    layer_name = 'conv2d_127'

    RANGER,CLASSES = add_ranger_classes_to_model(yolo_faulty.yolo_model,layer_name,NUM_INJECTIONS=8)
    yolo_ranger = RANGER.get_model()
    yolo_ranger.summary()
    layer = CLASSES_HELPER.get_layer(yolo_ranger,"classes_" + layer_name)
    layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
    
    gen1 = data_generator_wrapper('./../../keras-yolo3/pistols_dataset/export/',lines, 32, input_shape, anchors, num_classes)
    yolo_faulty.yolo_model = yolo_ranger

    for _ in range(7):
        dataset = next(gen1)
        data   = dataset[0][0]
        boxes  = dataset[2]
        labels_1 = dataset[0][1]
        labels_2 = dataset[0][2]
        labels_3 = dataset[0][3]

        from PIL import Image, ImageFont, ImageDraw
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)

    while True:
        print("loading Data")
        dataset = next(gen)
        data   = dataset[0][0]
        boxes  = dataset[2]
        labels_1 = dataset[0][1]
        labels_2 = dataset[0][2]
        labels_3 = dataset[0][3]

        from PIL import Image, ImageFont, ImageDraw
        
        img = np.uint8(data[0]*255)
        img = Image.fromarray(img)
        img.show()
        ex = input('press a key')
        r_image         = yolo.detect_image(img,y_true=boxes)#[labels_1,labels_2,labels_3]
        r_image_faulty  = yolo_faulty.detect_image(img,y_true=boxes)
        r_image.show()
        r_image_faulty.show()
        ex = input('press a key')
        if ex == 'q':
            exit()
    exit()
    yolo_out = yolo_model.predict(data)
    
    argss = [yolo_out[0],yolo_out[1],yolo_out[2],labels_1,labels_2,labels_3]

    print(data.shape)
    print(labels_1.shape)
    print(labels_2.shape)
    print(labels_3.shape)
    #args = {'yolo_outputs':yolo_out,'y_true':labels}
    loss = yolo_loss(argss,anchors,num_classes,0.5,custom_input_format=False)
    print(f"LOSS = [{loss}]")
    




if __name__ == '__main__':
    _main()

    
