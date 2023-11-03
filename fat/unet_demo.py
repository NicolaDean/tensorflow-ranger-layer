from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf
from enum import Enum
import sys
import os
import pathlib
from tensorflow.keras.utils import img_to_array, array_to_img
from tqdm import tqdm

# directory reach
LIBRARY_PATH = "./../"
sys.path.append(LIBRARY_PATH)
from model_helper.run_experiment import *
from our_datasets import pet
from our_datasets import self_drive

#2 types of errors in object detection setting:
# 1) number of boxes predicted -> predicted number must be equal to real one (binary evaluation: EQUAL or NOT EQUAL)
# 2) correspondance 1 to 1 for each box -> match is done looking at max iou index (binary evaluation can be done setting a threshold for stating whether
# the boxes actually match or not)

Error_ID_report = make_dataclass("Error_ID_report", 
                                     [("Set", str), ("Layer_name", str), ("Sample_id", int), ("Cardinality", int), ("Pattern", int), 
                                      ("Golden_num_boxes", int), ("Faulty_num_boxes", int), 
                                      ("Precision", float), ("Recall", float), ("F1_score", float),
                                      ("True_positives", float), ("False_positives", float), ("False_negatives", float), ("Error", str)])
#("Num_excluded",int),("Num_empty")
F1_score_report = make_dataclass("F1_score_report",[("Layer_name",str),("Epoch",int),("Num_wrong_box_shape",int),("Num_wrong_box_count",int),("TOT_Num_Misclassification",int),
                                                        ("Robustness",float),("V_F1_score",float),("I_F1_score",float),("G_F1_score",float),("V_accuracy",float),("I_accuracy",float),("G_accuracy",float),
                                                        ("V_precision",float),("I_precision",float),("G_precision",float),("V_recall",float),("I_recall",float),("G_recall",float),])
    

 #absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
def compute_rgb_similarity(a,b,th=5e-2,r_tol=0.01,a_tol=0.1):

    diff        = np.abs(a - b)
    threshold   = diff <= th
    num_pixel   = np.sum(np.ones_like(a))
    
    return np.sum(threshold) / num_pixel
    
    #Count how many similar pixel in the images
    gold_score = np.sum(np.isclose(a,b,rtol=0.01, atol=0.1))
    #Count num of pixels in general
    num_pixel  = np.sum(np.ones_like(a))
    
    #Percentage of similar pixels
    return gold_score/num_pixel
    
def create_mask(pred_mask,idx=None):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    if idx == None:
        return pred_mask
    else:
        return pred_mask[idx]

def calculate_iou(gt_mask, pred_mask, class_check=1,threshold = True):

    pred_mask = tf.equal(pred_mask , class_check)
    gt_mask   = tf.equal(gt_mask , class_check)

    overlap = tf.cast(tf.math.logical_and(pred_mask,gt_mask),dtype="float32")
    union   = tf.cast(tf.math.logical_or(pred_mask,gt_mask),dtype="float32")

    iou = tf.math.reduce_sum(overlap) / tf.math.reduce_sum(union)
    return iou

def post_fat_segmentation_report(injection_point,version="RGB"):


    if version == "CHANNEL_WISE":
        model = tf.keras.models.load_model("../saved_models/unet_self_drive_v2")
    else:
        model = tf.keras.models.load_model("../saved_models/unet_self_drive")

    model.summary()

    if version == "CHANNEL_WISE":
        train_generator_fn,place_holder = self_drive.get_generator(batch_size=32,not_valid=True)
        place_holder,val_generator_fn   = self_drive.get_generator(batch_size=1,not_train=True)

        train_size = 369
        valid_size = 100

        train_generator_fn = train_generator_fn()
        val_generator_fn   = val_generator_fn()

        #RAGE TUNE THE YOLO MODEL
        def range_tune(RANGER):
            print("=============FINE TUNING=============")
            for idx in tqdm(range(int(train_size//32))):
                    x_t,y_t = next(train_generator_fn)
                    RANGER.tune_model_range(x_t, reset=False,verbose=False)

    elif version == "RGB":
        x_train,y_train,x_val,y_val = self_drive.loadData(shape=256)
    
        #RAGE TUNE THE YOLO MODEL
        def range_tune(RANGER):
            print("=============FINE TUNING=============")
            #RANGER.tune_model_range(x_train, reset=False,verbose=True)

    print("PLOTTING:")
    if version == "CHANNEL_WISE":
        img,label = next(val_generator_fn)
    else:
        img     = np.expand_dims(x_val[0],0)
        label   = np.expand_dims(y_val[0],0)



    print("Layers on which we inject faults: ", str(injection_point))
    #if type(a_list) == list:
    RANGER,CLASSES = add_ranger_classes_to_model(model,[injection_point],NUM_INJECTIONS=60,use_classes_ranging=True,range_tuning_fn=range_tune,verbose=True)
    inj_model = RANGER.get_model()
    #yolo_ranger.summary()
    CLASSES.set_model(inj_model)
    CLASSES.disable_all(verbose=False)

    
    #inj_model.summary()
    
    ########################## REPORT #########################

    #Inizialize experiment variables
    print("-------------------------------")
    print(f'Injection on layer {injection_point}')
    print("-------------------------------")



    layer = CLASSES_HELPER.get_layer(inj_model,"classes_" + injection_point,verbose=False)
    assert isinstance(layer, ErrorSimulator)
    layer.set_mode(ErrorSimulatorMode.enabled)

    idx = 0

    print("PLOTTING:")
    if version == "CHANNEL_WISE":
        img,label = next(val_generator_fn)
    else:
        img     = np.expand_dims(x_val[idx],0)
        label   = np.expand_dims(y_val[idx],0)
    
    count = 0
    tot   = 0

    THRESHOLD   = [5e-2,2.5e-2,1e-2]
    THR_IDX     = 0
    GOOD_SCORE  = 0.7
    while 1:
        
        if version == "RGB":
            golden_pred, actual, mask     = self_drive.predict(img,label, model)
            injection_pred , actual, mask = self_drive.predict(img,label, inj_model)
            
            #absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`)) 
            gold_score = compute_rgb_similarity(golden_pred,mask,th=THRESHOLD[THR_IDX])
            print(f'GOLD VS GT SCORE = [{gold_score}]')

            inj_score = compute_rgb_similarity(injection_pred,golden_pred,th=THRESHOLD[THR_IDX])
            print(f'INJ VS GOLD SCORE = [{inj_score}]')

            ig_score = compute_rgb_similarity(injection_pred,mask,th=THRESHOLD[THR_IDX])
            print(f'INJ VS LAB SCORE = [{ig_score}]')
            
            final_frame = np.concatenate((actual[0],mask[0], golden_pred[0], injection_pred[0]), axis=1)

            tot += 1
            if inj_score < GOOD_SCORE:
                count += 1
            '''
            cv2.putText(final_frame, text=f"CURR THRESHOLD = [{THRESHOLD[THR_IDX]}] with GOOD_SCR = [{GOOD_SCORE}]", org=(6, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 100), thickness=2)
            
            cv2.putText(final_frame, text=f"SIMILARITY = [{inj_score}]", org=(6, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 100), thickness=2)
            
            cv2.putText(final_frame, text=f"MISC = [{count}] on total of [{tot}]", org=(6, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 100), thickness=2)
            
            cv2.putText(final_frame, text=f"MISC = [{count/tot}]", org=(6, 210), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 100), thickness=2)
            '''
        else:
            golden_pred     = model.predict(img)
            injection_pred  = inj_model.predict(img)
        
            rgb_label = np.apply_along_axis(self_drive.map_class_to_rgb, -1,label[0])*1./255
            rgb_gold  = np.apply_along_axis(self_drive.map_class_to_rgb, -1,golden_pred[0])*1./255
            rgb_inj   = np.apply_along_axis(self_drive.map_class_to_rgb, -1,injection_pred[0])*1./255


            final_frame = np.concatenate((img[0],rgb_label, rgb_gold, rgb_inj), axis=1)
        cv2.imshow("result", final_frame)

        k =  cv2.waitKey(1) & 0xFF
            
        if k == ord('q'):
            exit()
        elif k == ord('n'):
            if version=="CHANNEL_WISE":
                img,label = next(val_generator_fn)
            else:
                idx += 1

                if idx > len(x_val):
                    idx = 0

                img     = np.expand_dims(x_val[idx],0)
                label   = np.expand_dims(y_val[idx],0)
            count = 0
            tot   = 0  
        elif k == ord('t'):
            THR_IDX += 1
            if THR_IDX >= len(THRESHOLD):
                THR_IDX = 0 
            count = 0
            tot   = 0   
        elif k == ord('s'):
            GOOD_SCORE += 0.1
            if GOOD_SCORE > 0.9:
                GOOD_SCORE = 0.7

        #self_drive.Plotter(actuals[idx], golden_pred[idx], injection_pred[idx])
    
    exit()
    for image, mask in zip(x_val,y_val) :
        
        #Vanilla Model
        #CLASSES.disable_all(verbose=False)
        gold_predict = model.predict(image)
        aaaa         = inj_model.predict(image)

        idx = 0
        gold_predict = create_mask(gold_predict,idx)
        aaaa = create_mask(aaaa,idx)
        
        iou_0 = calculate_iou(mask[idx],gold_predict,0)
        iou_1 = calculate_iou(mask[idx],gold_predict,1)
        iou_2 = calculate_iou(mask[idx],gold_predict,2)

        mean  = (iou_0 + iou_1 + iou_2) / 3 
    
        print(f'GOLDEN IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}]')


        iou_0 = calculate_iou(mask[idx],aaaa,0)
        iou_1 = calculate_iou(mask[idx],aaaa,1)
        iou_2 = calculate_iou(mask[idx],aaaa,2)

        mean  = (iou_0 + iou_1 + iou_2) / 3 
    
        print(f'INJ IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}]')

        '''
        #pet.display([image[0], mask[0], create_mask(pred_mask,0)])
        #INJECTION:
        #CLASSES.disable_all(verbose=False)
        #layer = CLASSES_HELPER.get_layer(inj_model,"classes_" + injection_point,verbose=False)
        #assert isinstance(layer, ErrorSimulator)
        #layer.set_mode(ErrorSimulatorMode.enabled)

        print(image[0].shape)

        batched_inj   = tf.stack([image[0]] * 64)
        golden_label  = mask[0]#tf.stack([mask[0]] * 64)
        
        print(golden_label.shape)
        inj_pred = inj_model.predict(image)

        inj_pred = create_mask(inj_pred)
        
        iou_0 = calculate_iou(golden_label,inj_pred,0)
        iou_1 = calculate_iou(golden_label,inj_pred,1)
        iou_2 = calculate_iou(golden_label,inj_pred,2)
        mean  = (iou_0 + iou_1 + iou_2) / 3 

        print(f'INJECTION IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}]')

        #pet.display([gold_predict, golden_label[2], inj_pred[2]])

        '''

post_fat_segmentation_report(injection_point='conv2d_4')



'''
pred_mask = model.predict(image)
        tot_iou = 0
        for idx in range(0,64):
            #idx = 
            iou_0 = calculate_iou(mask[idx],create_mask(pred_mask,idx),0)
            iou_1 = calculate_iou(mask[idx],create_mask(pred_mask,idx),1)
            iou_2 = calculate_iou(mask[idx],create_mask(pred_mask,idx),2)

            tot_iou += iou_0 + iou_1 + iou_2

            mean  = (iou_0 + iou_1 + iou_2) / 3 
            print(f'IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}]')

        mean_incr = tot_iou / (64*3)

        iou_0 = calculate_iou(mask,create_mask(pred_mask),0)
        iou_1 = calculate_iou(mask,create_mask(pred_mask),1)
        iou_2 = calculate_iou(mask,create_mask(pred_mask),2)
        mean  = (iou_0 + iou_1 + iou_2) / 3 
        print(f'TOT IOU: [{iou_0}] ,\t [{iou_1}] ,\t [{iou_2}] \t=> MEAN [{mean}] vs [{mean_incr}]')

'''