from utils.ssd_model.load_model import load_model_ssd,load_image_into_numpy_array
from TFOD_loader.models.research.object_detection.utils import label_map_util
from TFOD_loader.models.research.object_detection.utils import visualization_utils as viz_utils

from tqdm import tqdm
from post_fat_report_ssd import get_vanilla_generator, classes_path,anchors_path,input_shape
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import io
import scipy.misc
import glob
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import sys
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

sys.path.append("./")

injection_points =  []
injection_points += ["block_3_project"]

model,CLASSES,RANGER,detect_fn,configs,vanilla_backone,inj_backbone = load_model_ssd(use_classes=True,injection_points=injection_points,dataset="pedestrian")

golden_gen_ranger,ranger_size   = get_vanilla_generator('./../../keras-yolo3/train/',32,classes_path,anchors_path,input_shape,random=False, keep_label=True, proc_img=False)
golden_gen_ranger,ranger_size   = get_vanilla_generator('./Self-Driving-Car-3/train/',32,classes_path,anchors_path,input_shape,random=False, keep_label=True, proc_img=False)

#Range Tune the model
#RAGE TUNE THE YOLO MODEL
print("=============FINE TUNING=============")
for _ in tqdm(range(ranger_size//32)):
        dataset = next(golden_gen_ranger)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)



#map labels for inference decoding
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

#run detector on test image
#it takes a little longer on the first run and then runs at normal speed.
import random

TEST_IMAGE_PATHS = glob.glob('./test/*.jpg')
TEST_IMAGE_PATHS = glob.glob('./Self-Driving-Car-3/valid/*.jpg')
image_path = random.choice(TEST_IMAGE_PATHS)
image_np = load_image_into_numpy_array(image_path)
import cv2
import tensorflow as tf

injection_status = True
while 1:
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    image_np_with_detections = image_np.copy()
    
    if injection_status:
        cv2.putText(image_np_with_detections, text="Injection ON : Classes", org=(6, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        layer = CLASSES_HELPER.get_layer(model._feature_extractor.classification_backbone,"classes_" + injection_points[0],verbose=False)
        assert isinstance(layer, ErrorSimulator)
        layer.set_mode(ErrorSimulatorMode.enabled)
    else:
        cv2.putText(image_np_with_detections, text="Injection OFF : Vanilla", org=(6, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        CLASSES.disable_all()

    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.5,
        agnostic_mode=False,
    )

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", image_np_with_detections)

    k =  cv2.waitKey(1) & 0xFF
        
    if k == ord('q'):
        #EXIT
        break
    elif k == ord('n'):
        image_path = random.choice(TEST_IMAGE_PATHS)
        image_np = load_image_into_numpy_array(image_path)
    elif k == ord('i'):
         injection_status = not injection_status
         
    