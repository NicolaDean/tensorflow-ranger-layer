import tensorflow as tf

import sys

sys.path.append("./../../")
from MobilenetSSD.load_model import SSD_MODEL

import pathlib
directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory +  "/../../")
from model_helper.run_experiment import *


directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + "/models/research")
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils


'''
conda install conda-build
conda-develop /home/nicola/tesi/MobilenetSSD
'''

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes   = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections      = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

#TODO => RIMETTERE I CHECKPOINT IN get_inference_model ... perchè non li carica?
def load_model_ssd(use_classes=False,injection_points=[]):
    #TODO => SET DATASET FROM HERE SO WE HAVE AN EASY TO USE TRAINING/INFERENCE POINT
    MODEL = SSD_MODEL(path="./mobilenet_ssd_config",model_name="ssd-mobilenet")

    #DOWNLOAD all the configurations and dataset and checkpoits necessary for the model
    MODEL.load_model()
    #Get a working model ready to execute inference
    most_recent_checkpoint_path                       = MODEL.get_most_recent_checkpoint_from_path('./ssd_training/')
    model,ssd_inference_fn,configs, model_config      = MODEL.get_inference_model(most_recent_checkpoint_path)

    # Run model through a dummy image so that variables are created
    image, shapes   = model.preprocess(tf.zeros([1, 320, 320, 3]))
    prediction_dict = model.predict(image, shapes)

    #Add classes and ranger
    vanilla_backone = model._feature_extractor.classification_backbone
    inj_backbone    = None

    if use_classes:
        RANGER,CLASSES = add_ranger_classes_to_model(vanilla_backone,injection_points,NUM_INJECTIONS=50)
        inj_backbone = RANGER.get_model()
        #yolo_ranger.summary()
        
        CLASSES.set_model(inj_backbone)
        CLASSES.disable_all()
        #TODO DEEP COPY THE MODEL SO TO HAVE A VANILLA COPY OF THE MODEL
        model._feature_extractor.classification_backbone = inj_backbone
    else:
        CLASSES = None
        RANGER  = None

    detect_fn = get_model_detection_function(model)
    return model,CLASSES,RANGER,detect_fn,configs,vanilla_backone,inj_backbone

    #Funziona solo dopo la prima inferenza dato che la backbone viene generata in fase di build del modello
    #model._feature_extractor.classification_backbone.summary()