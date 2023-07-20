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

layer_name = "conv_1"

yolov3 = load_yolo_with_weights()

RANGER,CLASSES = add_ranger_classes_to_model(yolov3,layer_name,NUM_INJECTIONS=128)
yolo_faulty = RANGER.get_model()
#layer = CLASSES_HELPER.get_layer(yolo_faulty,"classes_" + layer_name)
#layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
keras.utils.plot_model(yolo_faulty,to_file="yolo_classes.png" ,show_shapes=True)
keras.utils.plot_model(yolov3,to_file="yolo.png" ,show_shapes=True)

'''
yolov3.summary()
input("AAA")
yolo_faulty.summary()
'''

print(yolov3.output_names)
print(yolo_faulty.output_names)

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
        count += 1 # i.e. at 30 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        if count > 100:
            break

print("Ended Range Tuning")
RANGER.set_ranger_mode(RangerModes.Disabled)
RANGER.set_ranger_mode(RangerModes.Inference,RangerPolicies.Clipper,RangerGranularity.Layer)
layer = CLASSES_HELPER.get_layer(yolo_faulty,"classes_" + layer_name)
layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point

 # used to record the time when we processed last frame
prev_frame_time = 0
new_frame_time = 0
count = 1000
cap.set(cv2.CAP_PROP_POS_FRAMES, count)
# Read until video is completed
while(cap.isOpened()):
      
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        f1 = frame
        f2 = copy.deepcopy(f1)
        print("----------FAAAULT---------")
        aa = yolo_predict(yolo_faulty,f1)
        print("---------VANILLA----------")
        bb = yolo_predict(yolov3,f2)

        #print(tf.math.equal(aa[0],bb[0]))
        #print(tf.math.equal(aa[1],bb[1]))
        #print(tf.math.equal(aa[2],bb[2]))

        #print(aa[1] - bb[1])

        final_frame = np.concatenate((f1, f2), axis=1)

		# Display the resulting frame
        cv2.imshow('Frame', final_frame)
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        if cv2.waitKey(25) & 0xFF == ord('s'):
            count += 300 # i.e. at 30 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            print(f"SKIP TO {count} frame")
            break
        
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