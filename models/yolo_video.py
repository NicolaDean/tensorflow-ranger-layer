from yolo import *

# importing libraries
import cv2
import numpy as np
import time
import sys
import copy
# directory reach
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *

layer_name = "conv_100"

yolov3 = load_yolo_with_weights()

RANGER,CLASSES = add_ranger_classes_to_model(yolov3,layer_name,NUM_INJECTIONS=8)
yolo_faulty = RANGER.get_model()
layer = CLASSES_HELPER.get_layer(yolo_faulty,"classes_" + layer_name)
layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point


cap = cv2.VideoCapture('video.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

RANGER.set_ranger_mode(RangerModes.RangeTuning)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('video.mp4')
  
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")
CLASSES.disable_all() #Disable all fault injection points

count = 0
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        print(f"Frame : {count}")
        frame = cv2.resize(frame, (320,320), interpolation = cv2.INTER_AREA)
        frame = frame.astype('float32')
        frame /= 255.0
        frame = expand_dims(frame, 0)
        #TUNE THE LAYERS RANGE DOMAIN
        RANGER.tune_model_range(frame,reset=False)
        count += 30 # i.e. at 30 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        if count > 1000:
            break

print("AAAAAAAAAAAAAAAAAAAAA")
RANGER.set_ranger_mode(RangerModes.Disabled)
#RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Layer)
#layer = CLASSES_HELPER.get_layer(yolo_faulty,"classes_" + layer_name)
#layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point

 # used to record the time when we processed last frame
prev_frame_time = 0
new_frame_time = 0
count = 0
# Read until video is completed
while(cap.isOpened()):
      
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        f1 = frame
        f2 = copy.deepcopy(f1)
        yolo_predict(yolo_faulty,frame)
        #yolo_predict(yolov3,frame)

        final_frame = np.concatenate((f1, f2), axis=1)

		# Display the resulting frame
        cv2.imshow('Frame', frame)
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(25) & 0xFF == ord('s'):
            count += 300 # i.e. at 30 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            print(f"SKIP TO {count} frame")
            break
        else:
            count += 30 # i.e. at 30 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
  
# Break the loop
    else:
        break
  
# When everything done, release
# the video capture object
cap.release()
  
# Closes all the frames
cv2.destroyAllWindows()