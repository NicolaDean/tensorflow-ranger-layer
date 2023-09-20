
from .training.gen_golden_annotations import *
from .training.model_classes_init import *
from .training.ranger_helper import *

from .callbacks.random_injection import ClassesSingleLayerInjection
from .callbacks.metrics_obj import Obj_metrics_callback
from .callbacks.mixed_generator_v2 import MixedGeneratorV2Obj

import os
import shutil


#Declare path to dataset and hyperparameters
EXPERIMENT_NAME = "TEST"
FINAL_WEIGHT_NAME = "test.h5"
EPOCHS      = 1
batch_size  = 8
input_shape = (416,416) # multiple of 32, hw
annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'
annotation_path_valid   = './../../keras-yolo3/valid/_annotations.txt' 
classes_path            = './../../keras-yolo3/train/_classes.txt'         
anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt'
WEIGHT_FILE_PATH        = './../../keras-yolo3/yolo_boats_final.h5'

def init_path(EXPERIMENT_NAME=EXPERIMENT_NAME):
    root      = "./results/"
    log_dir   = root + EXPERIMENT_NAME 
    model_dir = root

    # remove old account directory
    shutil.rmtree(log_dir,ignore_errors=True)
    # create folders
    os.mkdir(log_dir)

    return root, log_dir, model_dir

#Declare list of injection layers 
'''
injection_points = ["conv2d", "batch_normalization"]
injection_points += ["conv2d_"+str(i) for i in range(1, 10)]
injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]
'''

injection_points = []


def run_fat_experiment(EPOCHS=EPOCHS,EXPERIMENT_NAME=EXPERIMENT_NAME,FINAL_WEIGHT_NAME=FINAL_WEIGHT_NAME,injection_points=injection_points,GOLDEN_LABEL = False, MIXED_LABEL = False, MIXED_LABEL_V2 = False, injection_frequency = 1.0, switch_prob = 0.5, num_epochs_switch = 1,custom_loss=False):

    root, log_dir, model_dir = init_path(EXPERIMENT_NAME)

    #Build a YOLO model with CLASSES and RANGER Integrated [TODO pass here the list of injection points]
    model, CLASSES, RANGER, vanilla_body,model_body = build_yolo_classes(WEIGHT_FILE_PATH,classes_path,anchors_path,input_shape,injection_points,classes_enable=True,custom_loss=custom_loss)

    #Get Dataset Generator
    if GOLDEN_LABEL:
        #Golden Labels
        golden_gen_train,train_size  = get_golden_generator(vanilla_body,'./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True)
        golden_gen_valid,valid_size  = get_golden_generator(vanilla_body,'./../../keras-yolo3/valid/',batch_size,classes_path,anchors_path,input_shape,random=True)
    else:
        #Classic Dataset Label
        golden_gen_train,train_size  = get_vanilla_generator('./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True, keep_label=False)
        golden_gen_valid,valid_size  = get_vanilla_generator('./../../keras-yolo3/valid/',batch_size,classes_path,anchors_path,input_shape,random=True, keep_label= False)

    #Tune ranger layers
    ranger_domain_tuning(RANGER,golden_gen_train,int(train_size/batch_size))

    #Declare injection point selection callback
    injection_layer_callback  = ClassesSingleLayerInjection(CLASSES,injection_points[0],extraction_frequency=injection_frequency,use_batch=True)
    reduce_lr                 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    f1_score                  = Obj_metrics_callback(model_body,'./../../keras-yolo3/valid/',classes_path,anchors_path,input_shape)

    checkpoint = ModelCheckpoint(log_dir +"/"+ EXPERIMENT_NAME + '-ep{epoch:03d}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=False, period=5)

    callbacks_list = [reduce_lr,f1_score,injection_layer_callback,checkpoint]

    if MIXED_LABEL:
        #Mixed labels 
        golden_gen_train, train_size = get_mixed_generator(vanilla_body, './../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True, switch_prob = switch_prob)
        golden_gen_valid,valid_size  = get_mixed_generator(vanilla_body,'./../../keras-yolo3/valid/',batch_size,classes_path,anchors_path,input_shape,random=True, switch_prob = switch_prob)
    elif MIXED_LABEL_V2:
        callback_obj = MixedGeneratorV2Obj(num_epochs_switch)
        golden_gen_train, train_size = get_mixed_v2_generator(vanilla_body, './../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True, callback_obj=callback_obj)
        callbacks_list.append(callback_obj)
    elif custom_loss:
        golden_gen_train, train_size = get_merged_generator(vanilla_body,'./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True)
        golden_gen_valid, valid_size = get_merged_generator(vanilla_body,'./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True)

    #Start training process
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_size, valid_size, batch_size))
    model.fit(golden_gen_train,
            steps_per_epoch=max(1, train_size//batch_size),
            validation_data=golden_gen_valid,
            validation_steps=max(1, valid_size//batch_size),
            epochs=EPOCHS,
            callbacks=callbacks_list)

    model_body.save_weights(model_dir + FINAL_WEIGHT_NAME,save_format='h5')
    loaded_model = model_body.load_weights(model_dir + FINAL_WEIGHT_NAME)

    #custom_objects={'Ranger':Ranger,'ErrorSimulator':ErrorSimulator}
    #model = load_model('temp.h5', custom_objects={'Conv2DIMC':Conv2DIMC}) 





