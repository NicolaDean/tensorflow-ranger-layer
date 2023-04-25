# CLASSES HELPER CLASS

This class aim to eliminate a good percentage of the boilerplate code necessary to work with classes

# Features
1. Allow to easily add ErrorSimulation layer to ALL Classes compatible Layers from existing vanilla models (Without any boileplate, handwritten code)
2. Allow to easily disable/enable a specific Fault Layer (TODO ADD THIS FEATURE ON ERROR SIMULATION LAYER TOO)
3. Generate a dictionary with "As a Function" functions for all Classes Compatible layers by supply a vanilla model
   
[**No boilerplate required, All those features generate code Automatically (See examples below)**]

# How to add The Error Simulator Layer automatically:

``` python
from classes_model import CLASSES_HELPER

model = load_model_from_file(...) #Load a model from keras or from file or create one

CLASSES = CLASSES_HELPER(model)   #Load the model into the Classes Helper
CLASSES.convert_model()           #Add ErrorSimulator Layer after every Classes Compatible Layer
CLASSES.disable_all_faults()      #Disable All Layers
CLASSES.enamble_fault_by_name("TargetLayerName")  #Change mode of Fault Layer to enamble by using the layer name
CLASSES.enamble_fault_by_name(1)#Change mode of Fault Layer by using target index

[...Your The Injection Campaing...]


```

# How to generate "As a Function" function automatically: