# CLASSES HELPER CLASS

This class aim to eliminate a good percentage of the boilerplate code necessary to work with classes

# Features
1. Allow to add Classes to each compatible layer in a all kind of models (eg: also Skip Connections one).
2. Easily Disable all classes layers.
3. Easily Enable a specific target layer by name.
4. Can also add classes to a restricted list of layers of your choice
5. Automated Fault injection Campaign that automatically enable injection points one at a time.
   
[**No boilerplate required, All those features generate code Automatically (See examples below)**]

# Add Classes Injector to ALL compatible layers:

``` python
from classes_model import CLASSES_HELPER

model = load_model_from_file(...) #Load a model from keras or from file or create one

num_requested_injection_sites = NUM_INJECTIONS * 5
#Load Model into Ranger Helper
CLASSES = CLASSES_HELPER(ranger_model)         

#Add Fault Injection Layer after each Convolutions or Maxpool
CLASSES.convert_model_v2(num_requested_injection_sites)

classes_model = CLASSES.get_model()
classes_model.summary()

#Disable all fault injection points
CLASSES.disable_all() 


[...Your The Injection Campaing...]


```
# Add Classes Injector to a restricted list of layers:
```python

from classes_model import CLASSES_HELPER


layer_names = ['conv2d_3','conv2d_4','conv2d_6','conv2d_9','conv2d_57','conv2d_60']

num_requested_injection_sites = NUM_INJECTIONS * 5
#Load Model into Ranger Helper
CLASSES = CLASSES_HELPER(ranger_model)

#Add Fault Injection Layer after each Convolutions or Maxpool
CLASSES.add_classes_by_name(layer_names,num_requested_injection_sites)

classes_model = CLASSES.get_model()
classes_model.summary()

#Disable all fault injection points
CLASSES.disable_all() 
```