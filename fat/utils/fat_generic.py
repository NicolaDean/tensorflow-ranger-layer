
from .callbacks.layer_selection_policy import ClassesLayerPolicy
from keras.callbacks import ReduceLROnPlateau


from .training.gen_golden_annotations import *
from .training.model_classes_init import *
from .training.ranger_helper import *


def run_fat_experiment(model,train_gen,valid_gen,train_size,valid_size,callbacks=[],frequency=0.5,epochs=20,injection_point=[],SKIP_INJECTION=False):
    
    print("BANANA")
    #RAGE TUNE THE YOLO MODEL
    def range_tune(RANGER):
            print("=============FINE TUNING=============")
            if not SKIP_INJECTION:
                    for idx in tqdm(range(int(train_size))):
                            x_t,y_t = next(train_gen)
                            RANGER.tune_model_range(x_t, reset=False,verbose=False)

    print("Layers on which we inject faults: ", str(injection_point))
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(model,injection_point,NUM_INJECTIONS=60,use_classes_ranging=True,range_tuning_fn=range_tune,verbose=True)
    inj_model = RANGER.get_model()
    #yolo_ranger.summary()
    CLASSES.set_model(inj_model)
    CLASSES.disable_all(verbose=False)

    injection_layer_callback  = ClassesLayerPolicy(CLASSES,extraction_frequency=frequency,use_batch=True,extraction_type=1)
    reduce_lr                 = ReduceLROnPlateau(monitor='va_loss', factor=0.1, patience=3, verbose=1, min_lr=0.000001)
    
    callbacks.append(injection_layer_callback)
    callbacks.append(reduce_lr)

    inj_model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

    hisotry=inj_model.fit(train_gen,
                validation_data=valid_gen,
                steps_per_epoch=train_size,
                validation_steps=valid_size,
                epochs=epochs,)

    inj_model.save_weights("../saved_models/unet_FAT_test.h5",save_format='h5')
    #inj_model.save("../saved_models/unet_FAT_test")