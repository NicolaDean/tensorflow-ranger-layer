from tensorflow import keras
import tensorflow as tf
import sys
import os
import pathlib

WEIGHT_FILE_PATH = "../saved_models/"
LIBRARY_PATH = "/../"


# directory reach
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + LIBRARY_PATH)
print("AAA:" + directory + LIBRARY_PATH)

from model_helper.ranger_model import *
from model_helper.classes_model import *
from models import *

model = keras.models.load_model("../saved_models/Dave2/dave_2.h5")

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(66,200,3), shuffle=True, path = "../datasets/steering_wheel/"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(k, self.list_IDs[k]) for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size)

        # Generate data
        for i, (I, ID) in enumerate(list_IDs_temp):
            # Store sample
            image = cv2.imread(self.path+str(ID)+".jpg")     #read images from disk
            image=cv2.resize(image[-150:], (200,66))/255  
            X[i,] = image
            
            # Store class
            y[i] = float(self.labels[I])

        return X, y

# Parameters for datagen.py
params = {'dim': (66,200,3),
          'batch_size': 64,
          'shuffle': True}


imgs_test = []
angles_test = []

with open("../datasets/steering_wheel/data.txt") as f:                                 #read steering angles from disk and preprocess
    lines = f.readlines()

for line in lines:
    data = line.split()
    for i in data:
        if i[-1]=='g':
            imgs_test.append(int(i[:-4]))
        else:
            angles_test.append(float(i))

imgs_train = []
angles_train = []

with open("../datasets/train_steering_wheel/data.txt") as f:                                 #read steering angles from disk and preprocess
    lines = f.readlines()

for line in lines:
    data = line.split()
    for i in data:
        if i[-1]=='g':
            imgs_train.append(int(i[:-4]))
        else:
            angles_train.append(float(i))



test_generator = DataGenerator(imgs_test, angles_test, **params)
train_generator = DataGenerator(imgs_train, angles_train, path = '../datasets/train_steering_wheel/', **params)

RANGER = RANGER_HELPER(model)
RANGER.convert_model()
ranger_model = RANGER.get_model()


RANGER.tune_model_range(train_generator)

NUM_INJECTIONS = 100

num_requested_injection_sites = NUM_INJECTIONS * 5
#Load Model into Ranger Helper
CLASSES = CLASSES_HELPER(ranger_model)         #PROBLEM HERE (??? TODO FIX ???) => With model work, with ranger_model not.. why??

#Add Fault Injection Layer after each Convolutions or Maxpool
CLASSES.convert_model(num_requested_injection_sites)
classes_model = CLASSES.get_model()
classes_model.summary()

CLASSES.disable_all() #Disable all fault injection points

RANGER.set_model(classes_model) #IMPORTANT (otherwise Ranger.set_ranger_mode would not work!)

x,y = test_generator.__getitem__(0)
RANGER.set_ranger_mode(RangerModes.Inference)


dataframe_row = make_dataclass("dataframe_row", [("layer_name",str),("label", float), ("vanilla", float), ("ranger", float)])

def eval_skew(y_batch,pred, label,layer_name):
    y_batch *= 180
    pred *= 180
    results = []
    for a, b, c in zip(y_batch, pred, label):
        a = np.squeeze(a)
        b = np.squeeze(b)
        results += [dataframe_row(layer_name,c,a,b)]
        print(f"Vanilla: {a}\t  Prediction: {b}\t Label: {c} DELTA {a-b}")
    
    return results

model.summary()

results = []
for layer_name in CLASSES.get_injection_points():
    for i in range(1):
        x, y = test_generator.__getitem__(i)
        CLASSES.disable_all() 
        layer = CLASSES_HELPER.get_layer(classes_model, layer_name)
        layer.set_mode(ErrorSimulatorMode.enabled)  #Enable the Selected Injection point
        vanilla = model.predict(x)
        ranger = classes_model.predict(x)
        results += eval_skew(vanilla, ranger, y,layer_name)

from dataclasses import make_dataclass

    


data = pd.DataFrame(results)
data.to_csv("report_steer_wheel.csv")#,
#vanilla = CLASSES.gen_model_injection_report(x,y,experiment_name = "FaultInjection",num_of_iteration=1, concat_previous=True, evaluation = eval_skew, ignore_misclassification=True)
