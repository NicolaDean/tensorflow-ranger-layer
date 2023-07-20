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
    yolo        = YOLO(**vars(argss))
    yolo_faulty = YOLO(**vars(argss))
    yolo_model = yolo.yolo_model

    yolo_model.summary()
    
    layer_names = ["conv2d_71"]
    #layer_names = ["conv2d", "batch_normalization"] 
    #layer_names += ["conv2d_"+str(i) for i in range(1, 10)]
    #layer_names += ["batch_normalization_"+str(i) for i in range(2, 10)]
    #layer_names += ["conv2d_25","conv2d_42","conv2d_56"]   #to remove
    #layer_names += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
    #layer_names += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]
    print("Layers on which we inject faults: ", str(layer_names))
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(yolo.yolo_model,layer_names,NUM_INJECTIONS=30)
    yolo_ranger = RANGER.get_model()
    yolo_ranger.summary()
    CLASSES.set_model(yolo_ranger)
    CLASSES.disable_all()

    yolo_faulty = yolo_ranger
   
    #RAGE TUNE THE YOLO MODEL
    print("=============FINE TUNING=============")
    for _ in range(12):
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
                                      ("IOU", float), ("Golden_num_boxes", int), ("Faulty_num_boxes", int), 
                                      ("Precision", float), ("Recall", float), ("F1_score", float),
                                      ("True_positives", float), ("False_positives", float), ("False_negatives", float), ("Error", str)])
    
    #report = pd.DataFrame(columns = Error_ID_report.__annotations__.keys())
    #report.to_csv("../reports/yolo_boats_test_NOrandom.csv")
    report = []

    from PIL import Image

    '''
    # VALIDATION SET

    #for all layers we want to inject faults
    for layer_name in layer_names:
        #for all test samples
        for sample_id in range(105):
            #load data
            dataset = next(valid_gen)
            data   = dataset[0][0]
            boxes  = dataset[2]
            labels_1 = dataset[0][1]
            labels_2 = dataset[0][2]
            labels_3 = dataset[0][3]
            img = np.uint8(data[0]*255)
            img = Image.fromarray(img)
            f1 = img
            f2 = copy.deepcopy(img)

            #vanilla prediction
            yolo.yolo_model = yolo_model
            r_image,v_out_boxes, v_out_scores, v_out_classes = yolo.detect_image(f1,y_true=False)
            r_image         = np.asarray(r_image)

            if v_out_boxes.shape[0] == 0:
                report += [Error_ID_report("valid", layer_name, sample_id, np.nan, np.nan, 
                                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                           "Golden prediction has no boxes")]
                continue
            
            #150 faults injections are performed
            yolo.yolo_model = yolo_faulty
            CLASSES.disable_all()
            layer = CLASSES_HELPER.get_layer(yolo_faulty,"classes_" + layer_name)
            assert isinstance(layer, ErrorSimulator)
            layer.set_mode(ErrorSimulatorMode.enabled)


            for _ in range(50):
                err = ""
                try:
                    r_image_faulty,out_boxes, out_scores, out_classes  = yolo.detect_image(f2,y_true=False, no_draw = True)
                    r_image_faulty  = np.asarray(r_image_faulty)
                except Exception as e:
                    err = e

                # get injected error id (cardinality, pattern)
                curr_error_id = layer.error_ids[layer.get_history()[-1]]
                curr_error_id = np.squeeze(curr_error_id)

                if err != "":
                    report += [Error_ID_report("valid", layer_name, sample_id, curr_error_id[0], curr_error_id[1], 
                                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, err)]
                    continue

                print("Boxes ", str(v_out_boxes))

                try:
                    iou_curr = compute_iou(v_out_boxes,v_out_classes,out_boxes, out_scores, out_classes).numpy()[0]
                except:
                    iou_curr = np.nan

                precision,recall,f1_score, TP, FP, FN = compute_F1_score(v_out_boxes,v_out_classes,out_boxes, out_classes, iou_th=0.5)

                report += [Error_ID_report("valid", layer_name, sample_id, curr_error_id[0], curr_error_id[1], 
                                           iou_curr, v_out_boxes.shape[0], out_boxes.shape[0], precision, recall, f1_score, TP, FP, FN, err)]
        report = pd.DataFrame(report)
        report.to_csv("../reports/yolo_boats_test_NOrandom.csv", mode = 'a', header = False)
        report = []
    '''         
    # TEST SET

    #for all layers we want to inject faults
    for layer_name in layer_names:
        #for all test samples
        for sample_id in range(32):
            #load data
            dataset = next(test_gen)
            data   = dataset[0][0]
            boxes  = dataset[2]
            labels_1 = dataset[0][1]
            labels_2 = dataset[0][2]
            labels_3 = dataset[0][3]
            img = np.uint8(data[0]*255)
            img = Image.fromarray(img)
            f1 = img
            f2 = copy.deepcopy(img)

            #vanilla prediction
            yolo.yolo_model = yolo_model
            r_image,v_out_boxes, v_out_scores, v_out_classes = yolo.detect_image(f1,y_true=False)
            r_image         = np.asarray(r_image)

            if v_out_boxes.shape[0] == 0:
                report += [Error_ID_report("test", layer_name, sample_id, np.nan, np.nan, 
                                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                           "Golden prediction has no boxes")]
                continue

            print("Boxes  ", str(v_out_boxes))
            
            #150 faults injections are performed
            yolo.yolo_model = yolo_faulty
            CLASSES.disable_all()
            layer = CLASSES_HELPER.get_layer(yolo_faulty,"classes_" + layer_name)
            assert isinstance(layer, ErrorSimulator)
            layer.set_mode(ErrorSimulatorMode.enabled)


            for _ in range(50):
                err = ""
                try:
                    r_image_faulty,out_boxes, out_scores, out_classes  = yolo.detect_image(f2,y_true=False, no_draw = True)
                    r_image_faulty  = np.asarray(r_image_faulty)
                except Exception as e:
                    err = e

                # get injected error id (cardinality, pattern)
                curr_error_id = layer.error_ids[layer.get_history()[-1]]
                curr_error_id = np.squeeze(curr_error_id)

                if err != "":
                    report += [Error_ID_report("test", layer_name, sample_id, curr_error_id[0], curr_error_id[1], 
                                           np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, err)]
                    continue

                try:
                    iou_curr = compute_iou(v_out_boxes,v_out_classes,out_boxes, out_scores, out_classes).numpy()[0]
                except:
                    iou_curr = np.nan
                precision,recall,f1_score, TP, FP, FN = compute_F1_score(v_out_boxes,v_out_classes,out_boxes, out_classes, iou_th=0.5)

                report += [Error_ID_report("test", layer_name, sample_id, curr_error_id[0], curr_error_id[1], 
                                           iou_curr, v_out_boxes.shape[0], out_boxes.shape[0], precision, recall, f1_score, TP, FP, FN, err)]
        report = pd.DataFrame(report)
        report.to_csv("../reports/yolo_boats_test_NOrandom.csv", mode = 'a', header = False)
        report = []        



    ###################### END REPORT ##########################


if __name__ == '__main__':
    _main()

    
