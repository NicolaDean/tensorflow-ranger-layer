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

#https://universe.roboflow.com/ds/ayvQYBRoRC?key=PFozYAMcyw => DATASET CALCIATORI


def enable_classes_layer(CLASSES,layer_name):
    '''
    Allow to enable a specific Classes injector point and disable all the others
    '''
    #Disable ALL layers
    CLASSES.disable_all()
    #Extract desired Classes injector
    layer = CLASSES_HELPER.get_layer(CLASSES.get_model(),"classes_" + layer_name)
    #Enable desired Injector
    layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
    
def _main():
    #Declare the Path to the necessary files
    annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'      #Annotations/Boxes info
    annotation_path_test    = './../../keras-yolo3/test/_annotations.txt'       
    classes_path            = './../../keras-yolo3/train/_classes.txt'          #List of classes of this problem
    anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt' #List of initial boxes config of yolo (keep default)
    
    anchors     = get_anchors(anchors_path)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")

    

    input_shape = (416,416) # multiple of 32, hw

    with open(annotation_path_train) as f:
        train_lines = f.readlines()
    
    with open(annotation_path_test) as f:
        test_lines = f.readlines()

    print(f"Found {len(train_lines)} elements in {annotation_path_train}")
    print(f"Found {len(test_lines)} elements in {annotation_path_test}")


    #Initialize the Data generator
    train_gen = data_generator_wrapper('./../../keras-yolo3/train/',train_lines, 32, input_shape, anchors, num_classes)
    test_gen  = data_generator_wrapper('./../../keras-yolo3/test/',test_lines, 1, input_shape, anchors, num_classes)

    
    #Necessary to pass arguments to the YOLO class (maybe there is better way)
    class args:
        def __init__ (self, model_path = './../../keras-yolo3/yolo_boats_final.h5',
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

    #DECLARE YOLO MODEL
    yolo        = YOLO(**vars(argss))
    yolo_model  = yolo.yolo_model
    yolo_model.summary()

    #DECLARE LIST OF INJECTION POINTS
    layer_names = ['conv2d_3','conv2d_4','conv2d_6','conv2d_9','conv2d_57','conv2d_60','conv2d_65'] #conv2d_21 è dannoso, conv2d_71 non tanto

    #Add to the Yolo model the list of desired injection points
    CLASSES     = add_classes_to_model(yolo_model,layer_names,NUM_INJECTIONS=8)
    yolo_model  = CLASSES.get_model()

    #With this function we can enable a specific layer by name
    enable_classes_layer(CLASSES,layer_names[0])


    #NOW WE TRY INJECT FAULTS ON INPUTS

    #Some Loop parameters 
    dataset = next(train_gen)
    n_inj = 1
    curr_injection = 0
    layer_injected_name = layer_names[curr_injection]
    iou_mean = 0
    update = True


    while True:
        print("loading Data")
        data   = dataset[0][0]
        boxes  = dataset[2]
        labels_1 = dataset[0][1]
        labels_2 = dataset[0][2]
        labels_3 = dataset[0][3]

        from PIL import Image
        
        #Convert the image into numpy array of uint8
        img = np.uint8(data[0]*255)
        #Convert image to PIL format so we can draw it to screen
        img = Image.fromarray(img) 

        #Predict boxes for this image
        r_image_faulty,out_boxes, out_scores, out_classes  = yolo.detect_image(img,y_true=False)
        r_image_faulty  = np.asarray(r_image_faulty)

        #Show image
        cv2.imshow("result", r_image_faulty)

        #Key pressed
        k =  cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):
            #EXIT
            break
        elif k == ord('i'):
            #CHANGE INJECTION POINT
            print("CHANGE INJECTION")
            curr_injection += 1

            if curr_injection >= len(layer_names):
                curr_injection = 0

            layer_injected_name = layer_names[curr_injection]
            enable_classes_layer(CLASSES,layer_injected_name)
        elif k == ord('d'):
            #DISABLE ALL FAULT INJECTION
            iou_mean = 0
            layer_injected_name = "Disabled"
            CLASSES.disable_all()
        elif k == ord('n'):
            #CHANGE TO NEXT INPUT IMAGE
            n_inj = 0
            dataset = next(train_gen)
        n_inj += 1


if __name__ == '__main__':
    _main()

    
