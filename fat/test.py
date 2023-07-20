from utils.training.gen_golden_annotations import *
from utils.training.model_classes_init import *
from train import *
from utils.callbacks.layer_selection_policy import ClassesLayerPolicy

#Declare path to dataset and hyperparameters
batch_size  = 32
input_shape = (416,416) # multiple of 32, hw
annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'
annotation_path_valid   = './../../keras-yolo3/valid/_annotations.txt' 
classes_path            = './../../keras-yolo3/train/_classes.txt'         
anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt'

#Declare list of injection layers 
injection_points = [""]

#Build a YOLO model with CLASSES and RANGER Integrated [TODO pass here the list of injection points]
model, CLASSES, RANGER = build_yolo_classes(classes_path,anchors_path,input_shape)

#Construct golden labels for train using robustness instead of accuracy
golden_gen_train,train_size = golden_generator(model,'./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True)
golden_gen_valid,valid_size = golden_generator(model,'./../../keras-yolo3/valid/',batch_size,classes_path,anchors_path,input_shape,random=True)

#Declare injection point selection callback
injection_layer_callback = ClassesLayerPolicy(injection_points,CLASSES)

#Declare object detection metrics callbacks

#Start training process
print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_size, valid_size, batch_size))
model.fit(golden_gen_train,
        steps_per_epoch=max(1, train_size//batch_size),
        validation_data=golden_gen_valid,
        validation_steps=max(1, valid_size//batch_size),
        epochs=100,
        callbacks=[injection_layer_callback])

#Save weights
model.save_weights('trained_weights_final.h5')