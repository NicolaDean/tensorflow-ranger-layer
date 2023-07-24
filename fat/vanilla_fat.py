
from utils.training.gen_golden_annotations import *
from utils.training.model_classes_init import *
from utils.training.ranger_helper import *

from utils.callbacks.layer_selection_policy import ClassesLayerPolicy
from utils.callbacks.metrics_obj import Obj_metrics_callback

#Declare path to dataset and hyperparameters
EPOCHS      = 20
batch_size  = 8
input_shape = (416,416) # multiple of 32, hw
annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'
annotation_path_valid   = './../../keras-yolo3/valid/_annotations.txt' 
classes_path            = './../../keras-yolo3/train/_classes.txt'         
anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt'
WEIGHT_FILE_PATH        = './../../keras-yolo3/yolo_boats_final.h5'

#Declare list of injection layers 
injection_points = ["conv2d", "batch_normalization"]
injection_points += ["conv2d_"+str(i) for i in range(1, 10)]
injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]

#Build a YOLO model with CLASSES and RANGER Integrated [TODO pass here the list of injection points]
model, CLASSES, RANGER, vanilla_body,model_body = build_yolo_classes(WEIGHT_FILE_PATH,classes_path,anchors_path,input_shape,injection_points,classes_enable=False)
#vanilla_body.summary()

golden_gen_train,train_size  = get_vanilla_generator('./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True)
golden_gen_valid,valid_size  = get_vanilla_generator('./../../keras-yolo3/valid/',batch_size,classes_path,anchors_path,input_shape,random=True)

#ranger_domain_tuning(RANGER,golden_gen_train,int(train_size/batch_size))

#Declare injection point selection callback
#injection_layer_callback  = ClassesLayerPolicy(CLASSES,uniform_extraction=True)
reduce_lr                 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
f1_score                  = Obj_metrics_callback(model_body,'./../../keras-yolo3/valid/',classes_path,anchors_path,input_shape)

callbacks_list = [reduce_lr,f1_score]

#Start training process
print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_size, valid_size, batch_size))
model.fit(golden_gen_train,
        steps_per_epoch=max(1, train_size//batch_size),
        validation_data=golden_gen_valid,
        validation_steps=max(1, valid_size//batch_size),
        epochs=EPOCHS,
        callbacks=callbacks_list)

#Save weights
model.save_weights('uniform_batch_v2.h5',save_format='h5')