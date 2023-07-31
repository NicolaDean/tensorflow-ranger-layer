sys.path.append("./../../../model_helper/")

from model_helper.classes_model import *

from f1_score import compute_f1_score
from ..training.gen_golden_annotations import *

def compute_f1_score():
    pass

def compute_model_robustness(model,injection_points,generator,gen_size):
    
    previous_inj_pt = injection_points[0]

    #Get vanilla predictions: (Create it apriori before training)

    for curr_injection_point in injection_points:
        #Disable previously selected injection point:
        layer = CLASSES_HELPER.get_layer(model,previous_inj_pt,verbose=False)
        layer.set_mode(ErrorSimulatorMode.disabled)  #Enable the Selected Injection point

        #Enable the selected injection point:
        layer = CLASSES_HELPER.get_layer(model,curr_injection_point,verbose=False)
        layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point

        for idx in tqdm(range(0,gen_size)):
            
            img, box_golden, classes_golden = next(golden_gen)

            yolo_out = model.predict(img,verbose=False)

            boxes, scores, classes = yolo_eval(yolo_out, self.anchors,self.num_classes, self.input_shape,score_threshold=0.5, iou_threshold=0.5)
            



