
from .load_model import load_model_ssd
import sys
import pathlib
directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory +  "/../../")
from model_helper.run_experiment import *
sys.path.append("./../")

from TFOD_loader import run_train
from utils.callbacks.layer_selection_policy import ClassesLayerPolicy

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

#TODO => AGGIUGNERE UNA LISTA SUBSET DI LAYER DA INIETTARE
def train_fat(injection_point,model,dataset_type,use_classes=False,callbacks=[],frequency=0.5,use_batch=False,range_tune=None):
    
    CONFIG_PATH     = f"./TFOD_models/{model}/{dataset_type}/fine_tuned_model"
    SAVE_MODEL_PATH = "./saved_TFOD_models"
    
    #ERRORE => CONFIG_PATH DOVREBBE ESSERE IL PATH AL MODELLO CONVERTITO DALLO SCRIPT SUL COLAB ("fine_tuned_model")
    #TODO => SET RANGES ON CLASSES
    
    #START THE ACTUAL TRAINING

    def modify_model(model):
        model._feature_extractor.classification_backbone.summary()
        if use_classes:
            vanilla_backone = model._feature_extractor.classification_backbone
            RANGER,CLASSES = add_ranger_classes_to_model(vanilla_backone,injection_point,NUM_INJECTIONS=50,use_classes_ranging=True,range_tuning_fn=range_tune)
            inj_backbone = RANGER.get_model()
            #yolo_ranger.summary()
            
            CLASSES.set_model(inj_backbone)
            CLASSES.disable_all()
            #TODO DEEP COPY THE MODEL SO TO HAVE A VANILLA COPY OF THE MODEL
            model._feature_extractor.classification_backbone = inj_backbone
            
        else:
            CLASSES = None
            RANGER  = None

        callbacks = []
        if use_classes:
            policy = ClassesLayerPolicy(CLASSES=CLASSES,extraction_frequency=frequency,use_batch=use_batch)
            policy.set_model(model._feature_extractor.classification_backbone)
            callbacks.append(policy)

        return model, callbacks
        detect_fn = get_model_detection_function(model)

    pipeline_file   = f'{CONFIG_PATH}/pipeline.config'
    model_dir       = SAVE_MODEL_PATH
    
    run_train(model_dir,pipeline_file,model=modify_model)
    
    #TODO => TRAIN_LOOP mettere un if model!=None usare questo invece di quello genrato da train_loop

    #SETTARE MODEL NELLE CALLBACK DI POLICY => model._feature_extractor.classification_backbone

    #SETTARE model come MODELLO DELLE ALTRE CALLBACK (F1,Mixed)
    
    #callback.set_model()



    '''
    def create_model():
        
        model,CLASSES,RANGER,detect_fn,configs,vanilla_backone,inj_backbone = load_model_ssd(use_classes=use_classes,injection_points=injection_point,dataset=dataset_type,config_path=CONFIG_PATH,range_tune=range_tune)
        
        callbacks = []
        if use_classes:
            policy = ClassesLayerPolicy(CLASSES=CLASSES,extraction_frequency=frequency,use_batch=use_batch)
            policy.set_model(model._feature_extractor.classification_backbone)
            callbacks.append(policy)

        return model, callbacks
    '''