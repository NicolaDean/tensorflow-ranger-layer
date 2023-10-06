
import random
from tqdm import tqdm
import sys
import numpy as np
import shutil

sys.path.append("./../../keras-yolo3/")
from yolo import YOLO, detect_video, custom_iou, compute_F1_score

SPLIT = 0.8

#LOAD ANNOTATIONS
annotation_path = './Self-Driving-Car-3/export/_annotations.txt'
input_shape = (416,416) # multiple of 32, hw

with open(annotation_path) as f:
    lines = f.readlines()
    random.shuffle(lines)

print(f"Found {len(lines)} elements in {annotation_path}")


def remove_box_duplicate(box):
    #FOR ALL BOX IN BOXES
    new_boxes = []
    b1 = 0
    num_removed = 0
    
    #print(f'Box Shape = {box.shape[0]}')
    if box.shape[0] == 1:
        return box,0

    while b1 < box.shape[0]:
        box_to_rem = []
        #FOR ALL OTHER BOX IN THE SAME LIST
        b2 = b1 +1
        while b2 < box.shape[0]:
            #Compute IOU
            iou = custom_iou(box[b1],box[b2])
            #IF DUPLICATED ADD TO LIST OF REMOVAL
            if iou > 0.90:
                box_to_rem.append(b2)
                num_removed += 1
            b2 +=1

        #Remove duplicated
        np.delete(box,box_to_rem)
        b1  += 1
    return box,num_removed

#PREPROCESSING:
new_lines = []
num_removed_tot = 0
for annotated_line in tqdm(lines):
    line = annotated_line.split()
    img_name = line[0]
    #print(line)
    #print(annotated_line)
    #print(line)
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    #MERGE TOGHETER ALL STOPLIGHT
    if box.shape[0] > 0:
        classes = box[:, 4]

        classes[classes == 4]   = 3
        classes[classes == 5]   = 3
        classes[classes == 6]   = 3
        classes[classes == 7]   = 3
        classes[classes == 8]   = 3
        classes[classes == 9]   = 3
        classes[classes == 10]  = 4
        
        box[:,4] = classes

        
    #CHECK FOR DUPLICATE (eg overlapped traffic light box)
    box,num_removed = remove_box_duplicate(box)

    num_removed_tot = num_removed_tot + num_removed
    #print(f"Removed: [{num_removed}] duplicates")
    
    #WRITE NEW LINE
    str_box = []
    for b in box:
        b = np.array(map(str, b))
        b = b.tolist()
        str_b = ",".join(b)
        str_box.append(str_b)
    
    final_str = img_name + " " +  " ".join(str_box)

    #print(final_str)

    #APPEND NEW LINE TO LINE LIST
    new_lines.append(final_str)

print(f'TOT REMOVED = [{num_removed_tot}]')

#VALIDATION - TRAINING SPLIT
print(f"SPLIT DATA IN VALIDATION AND TRAIN [{SPLIT}]")
SPLIT_INDEX = int(len(new_lines) * SPLIT)

train_lines = new_lines[:SPLIT_INDEX]
valid_lines = new_lines[SPLIT_INDEX:]

def move_files_to(lines,src_path,dst_path):
    import os
    for annotated_line in tqdm(lines):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        line = annotated_line.split()
        src_name= src_path  + line[0]
        dst_name= dst_path  + line[0]

        shutil.copyfile(src_name, dst_name)

        #Write line on file
        with open(dst_path + "_annotations.txt", 'a') as f:
            f.write(annotated_line + "\n")
            
    #Write class file
    with open(dst_path + "_classes.txt", 'a') as f:
        f.write("biker\n")
        f.write("car\n")
        f.write("pedestrian\n")
        f.write("trafficLight\n")
        f.write("truck\n")

#WRITE THE NEW ANNOTATIONS FILE
move_files_to(train_lines,"./Self-Driving-Car-3/export/","./Self-Driving-Car-3/train/")
move_files_to(valid_lines,"./Self-Driving-Car-3/export/","./Self-Driving-Car-3/valid/")





    

    
    