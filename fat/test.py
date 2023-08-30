from utils.training.gen_golden_annotations import *
from utils.training.model_classes_init import *

from utils.callbacks.layer_selection_policy import ClassesLayerPolicy
from utils.callbacks.metrics_obj import Obj_metrics_callback

#Declare path to dataset and hyperparameters
batch_size  = 32
input_shape = (416,416) # multiple of 32, hw
annotation_path_train   = './../../keras-yolo3/train/_annotations.txt'
annotation_path_valid   = './../../keras-yolo3/valid/_annotations.txt' 
classes_path            = './../../keras-yolo3/train/_classes.txt'         
anchors_path            = './../../keras-yolo3/model_data/yolo_anchors.txt'

#Declare list of injection layers 
injection_points = ["conv2d_25"]
#injection_points = ["conv2d", "batch_normalization"] 
#injection_points += ["conv2d_"+str(i) for i in range(1, 10)]
#injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
#injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
#injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]

#Build a YOLO model with CLASSES and RANGER Integrated [TODO pass here the list of injection points]
WEIGHT_FILE_PATH = './../../keras-yolo3/yolo_boats_final.h5'
model, CLASSES, RANGER, vanilla_body, yolo_ranger = build_yolo_classes(WEIGHT_FILE_PATH, classes_path,anchors_path,input_shape,injection_points,classes_enable=True)
#vanilla_body.summary()

#(32, 13, 13, 3, 10)
#Construct golden labels for train using robustness instead of accuracy
golden_gen_train,train_size = get_golden_generator(vanilla_body,'./../../keras-yolo3/train/',batch_size,classes_path,anchors_path,input_shape,random=True)
golden_gen_valid,valid_size = get_golden_generator(vanilla_body,'./../../keras-yolo3/valid/',batch_size,classes_path,anchors_path,input_shape,random=True)

#Declare injection point selection callback
injection_layer_callback = ClassesLayerPolicy(CLASSES)

#Declare object detection metrics callbacks
metric_callback = Obj_metrics_callback(vanilla_body, './../../keras-yolo3/valid/',classes_path,anchors_path,input_shape,model)
#https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/metrics/MeanAveragePrecisionMetric


callbacks = [injection_layer_callback, metric_callback]
#Start training process
print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_size, valid_size, batch_size))
model.fit(golden_gen_train,
        steps_per_epoch=max(1, train_size//batch_size),
        validation_data=golden_gen_valid,
        validation_steps=max(1, valid_size//batch_size),
        epochs=5,
        callbacks=[callbacks])

#Save weights
model.save_weights('trained_weights_final.h5')


'''
with open("./../../keras-yolo3/train/_annotations.txt") as f:
        annotation_lines = f.readlines()

class_names = get_classes(classes_path)
anchors     = get_anchors(anchors_path)
num_classes = len(class_names)

train_gen = data_generator_wrapper('./../../keras-yolo3/train/',annotation_lines, 32, input_shape, anchors, num_classes, random = False)
    
print("VANILLA")
result = next(train_gen)

print(result[0][0].shape)
print(result[0][1].shape)
print(result[0][2].shape)
print(result[0][3].shape)

print("GOLDEN")
result = next(golden_gen_valid)

print(result[0][0].shape)
print(result[0][1].shape)
print(result[0][2].shape)
print(result[0][3].shape)
exit()
'''


'''

with open("./../../keras-yolo3/train/_annotations.txt") as f:
        annotation_lines = f.readlines()

class_names = get_classes(classes_path)
anchors     = get_anchors(anchors_path)
num_classes = len(class_names)

train_gen = data_generator_wrapper('./../../keras-yolo3/train/',annotation_lines, 32, input_shape, anchors, num_classes, random = False)
    
print("VANILLA")
result = next(train_gen)

data   = result[0][0]
labels_1 = result[0][1]
labels_2 = result[0][2]
labels_3 = result[0][3]

print(data.shape)
print(labels_1.shape)
print(labels_2.shape)
print(labels_3.shape)

yolo_out = vanilla_body.predict(data)

print(yolo_out[0].shape)
print(yolo_out[1].shape)
print(yolo_out[2].shape)

yolo_out_0 = (np.reshape(yolo_out[0],(32,13,13,3,10)))
yolo_out_1 = (np.reshape(yolo_out[1],(32,26,26,3,10)))
yolo_out_2 = (np.reshape(yolo_out[2],(32,52,52,3,10)))


argss = [yolo_out[0],yolo_out[1],yolo_out[2],yolo_out_0,yolo_out_1,yolo_out_2]
#argss = [yolo_out[0],yolo_out[1],yolo_out[2],yolo_out[0],yolo_out[1],yolo_out[2]]

loss = yolo_loss(argss,anchors,num_classes,0.5,custom_input_format=False)
print(f'Loss = [{loss}]')

exit()
'''