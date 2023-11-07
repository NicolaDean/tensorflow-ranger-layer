import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
if len(gpu) != 0:
    print("LIMIT GPU GROWTH")
    tf.config.experimental.set_memory_growth(gpu[0], True) #limits gpu memory

from utils.ssd_model.train_fat import train_fat

from utils.training.gen_golden_annotations import *
from tqdm import tqdm

injection_points = ["block_16_expand"]

import os
    
DATASET     = "./../../keras-yolo3"
input_shape = (320,320)

classes_path           = f'{DATASET}/train/_classes.txt'
anchors_path           = './../../keras-yolo3/model_data/yolo_anchors.txt'


#IMPORTANT => SET THE proc_img to false to avoid YOLO preprocess on SSD model
golden_gen_ranger,ranger_size   = get_vanilla_generator(f'{DATASET}/train/',32,classes_path,anchors_path,input_shape,random=False, keep_label=True, proc_img=False)

def range_tune(RANGER):  
    print("=============FINE TUNING=============")
    if (ranger_size//32) > 100:
        size = 100
    else:
        size = ranger_size//32

    for _ in tqdm(range(size)):
        dataset = next(golden_gen_ranger)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)

train_fat(injection_points, 'aerial', use_classes=False,frequency=0,use_batch=True,range_tune=range_tune)