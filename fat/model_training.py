
from utils.training.gen_golden_annotations import *
from utils.training.model_classes_init import *
from utils.training.ranger_helper import *

from utils.callbacks.layer_selection_policy import ClassesLayerPolicy
from utils.callbacks.metrics_obj import Obj_metrics_callback

#Declare path to dataset and hyperparameters
EPOCHS                  = 300
EPOCHS_FINE_TUNING      = 200
MODEL_NAME              = "boats"

batch_size  = 16
input_shape = (416,416) # multiple of 32, hw
annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'
annotation_path_valid   = './../../keras-yolo3/valid/_annotations.txt' 
classes_path            = './../../keras-yolo3/train/_classes.txt'         
anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt'
WEIGHT_FILE_PATH        = './../../keras-yolo3/model_data/yolo.h5'

injection_points=[]
#Build a YOLO model with CLASSES and RANGER Integrated [TODO pass here the list of injection points]
model, CLASSES, RANGER, vanilla_body,model_body = build_yolo_classes(WEIGHT_FILE_PATH,classes_path,anchors_path,input_shape,injection_points,classes_enable=False,freeze_body=True)
#vanilla_body.summary()

golden_gen_train,train_size  = get_vanilla_generator('./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=False)
golden_gen_valid,valid_size  = get_vanilla_generator('./../../keras-yolo3/valid/',batch_size,classes_path,anchors_path,input_shape,random=False)

#Declare injection point selection callback
reduce_lr                 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
f1_score                  = Obj_metrics_callback(model_body,'./../../keras-yolo3/valid/',classes_path,anchors_path,input_shape,frequency=10)
early_stopping            = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

callbacks_list = [reduce_lr,f1_score,early_stopping]

#Start training process
print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_size, valid_size, batch_size))
model.fit(golden_gen_train,
        steps_per_epoch=max(1, train_size//batch_size),
        validation_data=golden_gen_valid,
        validation_steps=max(1, valid_size//batch_size),
        epochs=EPOCHS,
        callbacks=callbacks_list)


#Save weights
model.save_weights(MODEL_NAME + '_freezed.h5',save_format='h5')

for i in range(len(model.layers)):
        model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

#Start training process
print('FINE TUNING on {} samples, val on {} samples, with batch size {}.'.format(train_size, valid_size, batch_size))
model.fit(golden_gen_train,
        steps_per_epoch=max(1, train_size//batch_size),
        validation_data=golden_gen_valid,
        validation_steps=max(1, valid_size//batch_size),
        initial_epoch=EPOCHS,
        epochs=EPOCHS + EPOCHS_FINE_TUNING,
        callbacks=callbacks_list)

#Save weights
model.save_weights(MODEL_NAME + '_final.h5',save_format='h5')