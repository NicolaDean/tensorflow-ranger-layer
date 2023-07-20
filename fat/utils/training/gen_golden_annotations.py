import numpy as np
from PIL import Image

import sys
import pathlib
import os
directory = str(pathlib.Path(__file__).parent.parent.absolute()) +  "/../../../keras-yolo3"
sys.path.append(directory)

from yolo3.model import yolo_eval



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
            image_data = np.array(new_image)/255.

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        if len(box)>max_boxes: box = box[:max_boxes]
        box[:, [0,2]] = box[:, [0,2]]*scale + dx
        box[:, [1,3]] = box[:, [1,3]]*scale + dy
        box_data[:len(box)] = box

    return image_data, box_data

def generate_golden_annotations(model,data_gen,folder_path,annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    '''
    Inputs: 
    - Model: the model we use to generate the golden Labels
    - X: the Input images
    - Y: the True Labels of the image (Non serve in verità)

    Outputs: DataGenerator with golden prediction as label
    '''
    images          = []
    golden_labels   = []

    for sample in annotation_lines:
        #Extract image name
        image_name = sample.split(' ')[0] #Img file name
        #Load image from file
        image = load_image(folder_path,input_shape,sample)
        #Execute YOLO prediction
        yolo_out = model.predict(image)
        #Execute post processing of yolo result
        out_boxes, out_scores, out_classes = yolo_eval(yolo_out, anchors,num_classes, input_shape,score_threshold=0.7, iou_threshold=0.5)
        #Extract yolo golden label
        y_golden = np.column_stack(out_boxes,out_classes)

        images.append(image)
        golden_labels.append(y_golden)

    images          = np.array(images)
    golden_labels   = np.array(golden_labels)   
    
    return images,y_golden #An array of shape (Images,golden_annotations)

def golden_generator(model,folder_path,annotation_lines, batch_size, input_shape, anchors, num_classes, random):

    #Generate Golden Labels
    images, y_golden = generate_golden_annotations(model,folder_path,annotation_lines, batch_size, input_shape, anchors, num_classes, random)

    n = len(annotation_lines)
    i = 0
    #While app run
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if random:
                if i==0:
                    np.random.shuffle(annotation_lines)    

                image = images[i] #READ IMAGE (i)
                box   = y_golden[i]

                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)   

        yield [image_data, *box_data], np.zeros(batch_size)