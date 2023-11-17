from utils.training.gen_golden_annotations import *
import sys

sys.path.append("./../../keras-yolo3/")

from yolo import YOLO, detect_video, compute_iou, compute_F1_score
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from train import *

DATASET="./../../keras-yolo3"
DATASET="./Self-Driving-Car-3"
batch_size = 32
classes_path            = f'{DATASET}/train/_classes.txt'         
anchors_path            = f'./../../keras-yolo3/model_data/yolo_anchors.txt'

MODEL_WEIGHTS = "./../../keras-yolo3/yolo_boats_final.h5"
MODEL_WEIGHTS = "./results/pedestrian.h5"

class_names = get_classes(classes_path)
anchors     = get_anchors(anchors_path)
num_classes = len(class_names)

class args:
        def __init__ (self, model_path   = MODEL_WEIGHTS,
                            anchors_path = './../../keras-yolo3/model_data/yolo_anchors.txt',
                            classes_path =  classes_path,
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
model = yolo.yolo_model
input_shape = (416,416) # multiple of 32, hw

with open(f'{DATASET}/train/' + "_annotations.txt") as f:
        annotation_lines = f.readlines()

golden_gen_train,train_size  = generate_golden_annotations(model,f'{DATASET}/train/',annotation_lines,batch_size,input_shape,anchors,num_classes,random=True)
