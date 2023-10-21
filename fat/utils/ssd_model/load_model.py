import tensorflow as tf

import sys
from six import BytesIO
sys.path.append("./../../")
from TFOD_loader.load_model import SSD_MODEL

import pathlib
directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory +  "/../../")
from model_helper.run_experiment import *
'''
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + "/models/research")
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils
'''


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
def load_model_ssd(use_classes=False,injection_points=[],dataset='aerial'):
    
    if dataset == "aerial":
      #BOATS DATASET
      CHECKPOINTS_FOLDER = "./training" #Boats
      test_record_fname       = f'./Aerial-Maritime-9/test/movable-objects.tfrecord'
      train_record_fname      = f'./Aerial-Maritime-9/train/movable-objects.tfrecord'
      label_map_pbtxt_fname   = f'./Aerial-Maritime-9/train/movable-objects_label_map.pbtxt'
    elif dataset == "pedestrian":
      #SELF DRIVING PEDESTRIAN
      CHECKPOINTS_FOLDER      = "./pedestrian_ssd/training" #Pedestrian
      test_record_fname       = f'./Self-Driving-Car-3/train.tfrecord'
      train_record_fname      = f'./Self-Driving-Car-3/valid.tfrecord'
      label_map_pbtxt_fname   = f'./Self-Driving-Car-3/pedestrian_label_map.pbtxt'
    else:
       print(f"\033[0;31mDATASET {dataset} do NOT exits\033[0m")
      
    #TODO => SET DATASET FROM HERE SO WE HAVE AN EASY TO USE TRAINING/INFERENCE POINT
    MODEL = SSD_MODEL(path="./mobilenet_ssd_config",model_name="ssd-mobilenet")

    #DOWNLOAD all the configurations and dataset and checkpoits necessary for the model
    MODEL.load_model(train_record_fname,test_record_fname,label_map_pbtxt_fname)

    #Get a working model ready to execute inference
    most_recent_checkpoint_path                       = MODEL.get_most_recent_checkpoint_from_path(CHECKPOINTS_FOLDER)
    model,ssd_inference_fn,configs, model_config      = MODEL.get_inference_model(most_recent_checkpoint_path)

    model._feature_extractor.classification_backbone.summary()
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



def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)