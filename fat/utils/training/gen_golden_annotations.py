import numpy as np
from PIL import Image

import sys
import pathlib
import os

directory = str(pathlib.Path(__file__).parent.parent.absolute()) +  "/../../../keras-yolo3"
sys.path.append(directory)

from yolo3.model import yolo_eval,preprocess_true_boxes
from train1 import *


def get_vanilla_generator(folder_path, batch_size, classes_path,anchors_path,input_shape, random,keep_label=False):
    with open(folder_path + "_annotations.txt") as f:
        annotation_lines = f.readlines()

    class_names = get_classes(classes_path)
    anchors     = get_anchors(anchors_path)
    num_classes = len(class_names)

    train_gen = data_generator_wrapper(folder_path,annotation_lines, batch_size, input_shape, anchors, num_classes, random = False,keep_label=keep_label)
    
    return train_gen, len(annotation_lines)

def load_image(folder_path, input_shape, annotation_line,max_boxes=20, proc_img=True):
    line = annotation_line.split()
    img_name=folder_path + line[0]

    image = Image.open(img_name)
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    
    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0
    if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, dtype='float32')
            image_data /= 255.
    
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        if len(box)>max_boxes: box = box[:max_boxes]
        box[:, [0,2]] = box[:, [0,2]]*scale + dx
        box[:, [1,3]] = box[:, [1,3]]*scale + dy
        box_data[:len(box)] = box
    #print(f'True label = {box_data[0:3]}')
    
    return image_data, box_data

def generate_golden_annotations(model,folder_path,annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    '''
    Inputs: 
    - Model: the model we use to generate the golden Labels
    - X: the Input images
    - Y: the True Labels of the image (Non serve in verità)

    Outputs: DataGenerator with golden prediction as label
    '''
    images          = []
    golden_labels   = []

    from tqdm import tqdm
    for sample in tqdm(annotation_lines):
        #Extract image name
        image_name = sample.split(' ')[0] #Img file name
        #Load image from file
        image,label = load_image(folder_path,input_shape,sample)
        image_data = np.expand_dims(image, 0)
        #print(image.shape)
        #print(label.shape)
        #Execute YOLO prediction
        yolo_out = model.predict(image_data,verbose=0)
        #Execute post processing of yolo result
        out_boxes, out_scores, out_classes = yolo_eval(yolo_out, anchors,num_classes, input_shape,score_threshold=0.7, iou_threshold=0.5)
        #Extract yolo golden label
        
        y_golden = np.column_stack((out_boxes,out_classes))
        y_golden[:, [0, 1]] = y_golden[:, [1, 0]]
        y_golden[:, [2, 3]] = y_golden[:, [3, 2]]
        
        print(f'Golden labels {y_golden}')
        #print(y_golden)
        max_boxes = 20
        # correct boxes padding
        box_data = np.zeros((max_boxes,5))
        if len(y_golden)>0:
            np.random.shuffle(y_golden)
            if len(y_golden)>max_boxes: y_golden = y_golden[:max_boxes]
            box_data[:len(y_golden)] = y_golden
        
        images.append(image)
        golden_labels.append(box_data)

    images       = np.array(images)

    golden_labels   = np.array(golden_labels)   
    
    print("---------------------")
    print("Dataset Shape:")
    print("---------------------")
    print(images.shape)
    print(golden_labels.shape)
    print("---------------------")

    return images,golden_labels #An array of shape (Images,golden_annotations)

def golden_generator(model,folder_path, batch_size, classes_path,anchors_path,input_shape, random):

    with open(folder_path + "_annotations.txt") as f:
        annotation_lines = f.readlines()

    class_names = get_classes(classes_path)
    anchors     = get_anchors(anchors_path)
    num_classes = len(class_names)

    #Generate Golden Labels
    images, y_golden = generate_golden_annotations(model,folder_path,annotation_lines, batch_size, input_shape, anchors, num_classes, random=random)

    n = len(annotation_lines)
    i = 0
    #While app run
    while True:
        image_data      = []
        box_data        = []

        for b in range(batch_size):
            if random:
                if i==0:
                    #TODO SHUFFLE THE INPUT IMAGES INSTEAD TO MAKE SHUFFLE WORK
                    np.random.shuffle(annotation_lines)    
                    
                image  = images[i] #READ IMAGE (i)
                labels = y_golden[i]
                image_data.append(image)
                box_data.append(labels)
  
                i = (i+1) % n

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)
        '''
        yolo_out_0  = np.array(yolo_out_0)   
        yolo_out_1  = np.array(yolo_out_1)   
        yolo_out_2  = np.array(yolo_out_2) 

        golden_labels   = [yolo_out_0,yolo_out_1,yolo_out_2]

        print(golden_labels[0].shape)
        print(golden_labels[1].shape)
        print(golden_labels[2].shape)
        exit()
        '''
        
        golden_labels = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *golden_labels], np.zeros(batch_size)

def get_golden_generator(model,folder_path, batch_size, classes_path,anchors_path,input_shape, random):
    with open(folder_path + "_annotations.txt") as f:
        annotation_lines = f.readlines()
    
    gen =  golden_generator(model,folder_path, batch_size, classes_path,anchors_path,input_shape, random)
    return gen,len(annotation_lines)