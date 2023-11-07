
from .load_model import load_model_ssd
import sys
sys.path.append("./../../")
from model_helper.run_experiment import *

sys.path.append("./../")

from TFOD_loader import run_train
from utils.callbacks.layer_selection_policy import ClassesLayerPolicy

#TODO => AGGIUGNERE UNA LISTA SUBSET DI LAYER DA INIETTARE
def train_fat(injection_point,dataset_type,use_classes=False,callbacks=[],frequency=0.5,use_batch=False,range_tune=None):
    CONFIG_PATH     = "./mobilenet_ssd_config"
    SAVE_MODEL_PATH = "./saved_TFOD_models"
    
    #TODO => SET RANGES ON CLASSES
    
    #START THE ACTUAL TRAINING

    def modify_model(model):
        

    def create_model():
        
        model,CLASSES,RANGER,detect_fn,configs,vanilla_backone,inj_backbone = load_model_ssd(use_classes=use_classes,injection_points=injection_point,dataset=dataset_type,config_path=CONFIG_PATH,range_tune=range_tune)
        
        callbacks = []
        if use_classes:
            policy = ClassesLayerPolicy(CLASSES=CLASSES,extraction_frequency=frequency,use_batch=use_batch)
            policy.set_model(model._feature_extractor.classification_backbone)
            callbacks.append(policy)

        return model, callbacks
        
    pipeline_file   = f'{CONFIG_PATH}/pipeline_file.config'
    model_dir       = SAVE_MODEL_PATH
    
    run_train(model_dir,pipeline_file,model=None)
    
    #TODO => TRAIN_LOOP mettere un if model!=None usare questo invece di quello genrato da train_loop

    #SETTARE MODEL NELLE CALLBACK DI POLICY => model._feature_extractor.classification_backbone

    #SETTARE model come MODELLO DELLE ALTRE CALLBACK (F1,Mixed)
    
    #callback.set_model()

