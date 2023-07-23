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
injection_points = ["conv2d", "batch_normalization"] 

#Build a YOLO model with CLASSES and RANGER Integrated [TODO pass here the list of injection points]
model, CLASSES, RANGER, vanilla_body = build_yolo_classes(classes_path,anchors_path,input_shape,injection_points,classes_enable=False)
#vanilla_body.summary()

with open("./../../keras-yolo3/train/_annotations.txt") as f:
        annotation_lines = f.readlines()

class_names = get_classes(classes_path)
anchors     = get_anchors(anchors_path)
num_classes = len(class_names)

train_gen = data_generator_wrapper('./../../keras-yolo3/train/',annotation_lines, 1, input_shape, anchors, num_classes, random = False)
    

print("VANILLA")

for _ in range(0,10):
        result = next(train_gen)

        data     = result[0][0]
        labels_1 = result[0][1]
        labels_2 = result[0][2]
        labels_3 = result[0][3]


        labels = [labels_1,labels_2,labels_3]

        yolo_out = vanilla_body.predict(data)
        out_boxes, out_scores, out_classes = yolo_eval(yolo_out, anchors,num_classes, input_shape,score_threshold=0.7, iou_threshold=0.5)
        y_golden = np.column_stack((out_boxes,out_classes))
        #La label di yolo_eval sembra invertita tipo [ymin,xmin,ymax,xmax] invece di [xyxy] => DA VERIFICAREEE
        y_golden[:, [0, 1]] = y_golden[:, [1, 0]]
        y_golden[:, [2, 3]] = y_golden[:, [3, 2]]

        #print(y_golden)

        max_boxes = 5
        # correct boxes padding
        box_data = np.zeros((max_boxes,5))
        if len(y_golden)>0:
                np.random.shuffle(y_golden)
                if len(y_golden)>max_boxes: y_golden = y_golden[:max_boxes]
                box_data[:len(y_golden)] = y_golden

        golden_labels   = np.array([box_data]) 
        #print(golden_labels.shape)
        golden_labels   = preprocess_true_boxes(golden_labels, input_shape, anchors, num_classes) 
        #print(golden_labels[0].shape)

        yolo_out = vanilla_body.predict(data)

        argss = [*yolo_out,*golden_labels]
        loss = yolo_loss(argss,anchors,num_classes,0.5,custom_input_format=False)
        print(f'Loss = [{loss}]')

exit()
argss = [*yolo_out,*labels]
loss = yolo_loss(argss,anchors,num_classes,0.5,custom_input_format=False)
print(f'Loss = [{loss}]')

