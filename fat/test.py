from utils.training.gen_golden_annotations import *
from train import *


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

train_gen = data_generator_wrapper('./../../keras-yolo3/train/',train_lines, 32, input_shape, anchors, num_classes, random = False)

golden_gen = golden_generator()