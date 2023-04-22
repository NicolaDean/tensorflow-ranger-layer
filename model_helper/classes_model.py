from keras import datasets, layers, models, losses
from tensorflow import keras
import tensorflow as tf

class CLASSES_HELPER():
    def __init__(self,model:tf.keras.Model):
        self.model      = model #Model that we want to test.
        self.misc_mask  = None  #Contain a mask with 0 if vanilla model classify wrong, 1 if vanilla model classify ok

    '''
    Add an Injection point after layer with specified name
    '''
    def add_injections_from_names(self,layer_names):
        return
    
    '''
    Create a boolean vector with following propertys:
        V[i] = 0 <=> i-th image is misclassified by vanilla model
        V[i] = 1 <=> i-th image is correctly classified by vanilla model

    INPUTS: Dataset and Label
    '''
    def extract_misclassification_mask(self,X,Y):
        return

    '''
    Given a Dataset:
    For each injection point inside the model:
        Run an epoch
        Compute accuracy, Misclassification etc... [If exist, take mask in consideration, if not generate it automatically]
    Generate a detailed report in Dataframe format
    '''
    def gen_injection_report(self,X,Y):
        return


'''
IDEAS ON HOW IT SHOULD WORK

#------------MODEL DEFINITION---------------------------------------
input = InputLayer(...)

x = Conv2D(...)(input)
x = Conv2D(...)(x)
....
x = Conv2D(...)(x)
output = Dense(...)(x)

#-----COMPILE THE MODEL AND TRAIN (OR LOAD EXISTING MODEL---------
model = RangerModel(input, output, "My_ranger_Model")
model.compile(...)

#------------ADD INJECTION POINTS---------------------------------

CLASSES = CLASSES_HELPER(model)                #Load the Model into Classes helper

CLASSES.extract_misclassification_mask(X,Y)    #Extract a mask to cut out from evaluation the vanilla misclassification.

CLASSES.add_injections_from_names("conv1",[other args])
CLASSES.add_injections_from_names("conv2",[other args])
CLASSES.add_injections_from_names("conv3",[other args])
CLASSES.add_injections_from_names("maxpool1",[other args])


#------------GENERATE REPORT---------------------------------
report = CLASSES.gen_injection_report(X,Y)


#NOW WE HAVE A REPORT IN DATAFRAME FORMAT
'''