import tensorflow as tf
import keras


def vanilla_inference(self,logs):
    #Disable classes
    self.CLASSES.disable_all(verbose=False)
    #Get current batch info
    
    x_batch, y_batch = logs['inputs'], logs['targets']
    #Predict on golden model
    prediction       = self.model(x_batch)

    return prediction

class CustomLossV2VanillaPredictor(keras.callbacks.Callback):

    def __init__(self):
        self.curr_yolo_out  = None
        self.first_run        = tf.Variable(1,trainable=False   ,name="first_run")
        self.first_run.assign(tf.constant(1))

    def set_model_classes(self,CLASSES,yolo_body,input):
        self.CLASSES        = CLASSES
        self.model          = yolo_body
        self.curr_yolo_out  = self.model(input)
        self.first_run.assign(tf.constant(1))

    #Before batch start predict using golden model (No injection)
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(keys)
        self.first_run.assign(tf.constant(0))
        self.curr_yolo_out = vanilla_inference(self,logs)

    def on_test_batch_end(self, batch, logs=None):
        self.first_run.assign(tf.constant(0))
        self.curr_yolo_out = vanilla_inference(self,logs)