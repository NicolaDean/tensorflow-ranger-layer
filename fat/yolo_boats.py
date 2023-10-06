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
#https://public.roboflow.com/object-detection

def _main(checkpoint='./../../keras-yolo3/boats_final.h5',DATASET="./../../keras-yolo3"):
    # './export/_annotations.txt'
    annotation_path_train = f'{DATASET}/train/_annotations.txt'
    annotation_path_test  = f'{DATASET}/valid/_annotations.txt'
    #log_dir = 'logs/000/'
    # './export/_annotations.txt'
    classes_path = f'{DATASET}/train/_classes.txt'
    anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    print("-------------------CLASS NAMES-------------------")
    print(class_names)
    print("-------------------CLASS NAMES-------------------")
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    #Annotation File contain a row for each sample in format [FileName, [Bounding Boxes], Class]
    with open(annotation_path_train) as f:
        train_lines = f.readlines()
    
    with open(annotation_path_test) as f:
        test_lines = f.readlines()

    print(f"Found {len(train_lines)} elements in {annotation_path_train}")
    print(f"Found {len(test_lines)} elements in {annotation_path_test}")

    #We pass to a Data generator the list of available files in the annotation file
    train_gen = data_generator_wrapper(f'{DATASET}/train/',train_lines, 32, input_shape, anchors, num_classes)
    test_gen  = data_generator_wrapper(f'{DATASET}/valid/',test_lines, 1, input_shape, anchors, num_classes)

    #We create a fake args variable compatible with the YOLO class.
    class args:
        def __init__ (self, model_path = checkpoint,             #Put your weight file
                            anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt',   #Keep default
                            classes_path =  classes_path,   #Put your class file (each row is a label name)
                            score = 0.3,                    # 
                            iou = 0.45,                     #IOU threshold
                            model_image_size = (416, 416),  #Input image
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
    #Create the YOLO MODEL with the create args
    yolo        = YOLO(**vars(argss))   
    yolo_faulty = YOLO(**vars(argss))

    #We can get the keras model with parameter: yolo_model
    yolo_model = yolo.yolo_model        

    yolo_model.summary()
    #keras.utils.plot_model(yolo_model,to_file="yolomodel.png" ,show_shapes=True)
    #exit()
    #layer_names = ['conv2d_3','conv2d_4','conv2d_6','conv2d_9','conv2d_57','conv2d_60','conv2d_65'] #conv2d_21 è dannoso, conv2d_71 non tanto
    layer_names = []
    layer_names += ["batch_normalization_"+str(i) for i in range(3, 10)]
    layer_names += ["batch_normalization_"+str(i*10) for i in range(2, 8)]
    layer_names += ["conv2d_"+str(i) for i in range(3, 10)]
    layer_names += ["conv2d_"+str(i*10) for i in range(2, 8)]
    
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(yolo.yolo_model,layer_names,NUM_INJECTIONS=8)
    yolo_ranger = RANGER.get_model()
    yolo_ranger.summary()
    CLASSES.set_model(yolo_ranger)
    CLASSES.disable_all()

    yolo_faulty = yolo_ranger
   
    #RAGE TUNE THE YOLO MODEL
    print("=============FINE TUNING=============")
    for _ in range(5):
        dataset = next(train_gen)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)
    
    layer = CLASSES_HELPER.get_layer(yolo_ranger,"classes_" + layer_names[0])
    layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
    
    #NOW WE TRY INJECT FAULTS ON INPUTS
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
        
        img = np.uint8(data[0]*255)
        img = Image.fromarray(img)

        f1 = img
        f2 = copy.deepcopy(img)
        if update:
            #Vanilla
            yolo.yolo_model = yolo_model
            r_image,v_out_boxes, v_out_scores, v_out_classes = yolo.detect_image(f1,y_true=False)
            r_image         = np.asarray(r_image)
            update = False
        #Faulty
        yolo.yolo_model = yolo_faulty
        r_image_faulty,out_boxes, out_scores, out_classes  = yolo.detect_image(f2,y_true=False)
        r_image_faulty  = np.asarray(r_image_faulty)

        iou_curr = compute_iou(v_out_boxes,v_out_classes,out_boxes, out_scores, out_classes)
        res = compute_F1_score(v_out_boxes,v_out_classes,out_boxes, out_classes)
        precision,recall,f1_score, TP, FP, FN = res

        if iou_curr == None:
            iou_curr = 0
        iou_mean = ((iou_mean * (n_inj-1)) + iou_curr)/n_inj
        
        #ADD SOME TEXT TO THE PLOTTED IMAGE
        cv2.putText(r_image, text="Vanilla", org=(6, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.putText(r_image_faulty, text="Faulty", org=(6, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.putText(r_image_faulty, text=f"Num Injection = {n_inj}", org=(6, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.putText(r_image_faulty, text=f"Layer Injected = [{layer_injected_name}]", org=(6, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.putText(r_image_faulty, text=f"Current IOU = [{iou_curr}]", org=(6, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 50), thickness=2)
        cv2.putText(r_image_faulty, text=f"Mean    IOU = [{iou_mean}]", org=(6, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 50), thickness=2)
        cv2.putText(r_image_faulty, text=f"precison: [{precision}] , recall: [{recall}], f1_score: [{f1_score}]", org=(6, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 50), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)

        #Merge frame
        final_frame = np.concatenate((r_image, r_image_faulty), axis=1)
        cv2.imshow("result", final_frame)

        k =  cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):
            #EXIT
            break
        elif k == ord('i'):
            #CHANGE INJECTION POINT
            print("CHANGE INJECTION")
            iou_mean = 0
            n_inj = 0
            curr_injection += 1

            if curr_injection >= len(layer_names):
                curr_injection = 0

            layer_injected_name = layer_names[curr_injection]
            CLASSES.disable_all()
            layer = CLASSES_HELPER.get_layer(yolo_ranger,"classes_" + layer_injected_name)
            layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
        elif k == ord('d'):
            #DISABLE ALL FAULT INJECTION
            iou_mean = 0
            layer_injected_name = "Disabled"
            CLASSES.disable_all()
        elif k == ord('n'):
            #CHANGE TO NEXT INPUT IMAGE
            iou_mean = 0
            update = True
            n_inj = 0
            dataset = next(train_gen)
        n_inj += 1


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
    



import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default='./../../keras-yolo3/boats_final.h5', action = "store")
    parser.add_argument("--dataset",default="./../../keras-yolo3",action="store")

    args            = parser.parse_args()
    checkpoint      = str(args.checkpoint)
    DATASET         = str(args.dataset)

    _main(checkpoint=checkpoint,DATASET=DATASET)

    
