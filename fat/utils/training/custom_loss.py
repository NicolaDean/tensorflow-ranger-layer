import sys
import pathlib

directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory +  "/../../../keras-yolo3")

import tensorflow as tf
from yolo3.model import yolo_loss


def custom_loss_combinator(vanilla,golden):
    return golden + vanilla

def custom_yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False,custom_loss_combinator=custom_loss_combinator):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers      = len(anchors)//3 # default setting

    #GET YOLO OUTPUT
    yolo_outputs    = args[:num_layers]             # [0,1,2]

    #GET THE LABELS
    vanilla_out     = args[num_layers:num_layers*2] #[3,4,5]
    golden_out      = args[num_layers*2:]           #[6,7,8]

    #Define Loss args
    vanilla_args    = [*yolo_outputs, *vanilla_out]
    golden_args     = [*yolo_outputs, *golden_out ]

    #Compute the "local loss"
    vanilla_loss    = yolo_loss(vanilla_args,anchors, num_classes, ignore_thresh=ignore_thresh, print_loss=print_loss)
    golden_loss     = yolo_loss(golden_args ,anchors, num_classes, ignore_thresh=ignore_thresh, print_loss=print_loss)
    
    #Combine the loss function
    loss            = custom_loss_combinator(vanilla_loss,golden_loss)

    return loss


def custom_yolo_loss_v2(args, anchors, num_classes, ignore_thresh=.5, print_loss=False,custom_loss_combinator=custom_loss_combinator,custom_loss_callback=None,yolo_body=None,CLASSES=None):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    
    num_layers      = len(anchors)//3 # default setting
    
    #prediction without injection
    image_data      = args[num_layers*2]           #[6]

    #GET YOLO OUTPUT
    yolo_outputs_inj    = args[:num_layers]                     # [0,1,2]
    yolo_outputs_gold   = custom_loss_callback.curr_yolo_out    #golden pred generated by callback
    #print(yolo_outputs_gold[0].shape)

    #GET THE LABELS
    dataset_label       = args[num_layers:num_layers*2] #[3,4,5]
    
    #Define Loss args
    inj_args        = [*yolo_outputs_inj,  *dataset_label]
    golden_args     = [*yolo_outputs_gold, *dataset_label]

    #Compute the "local loss"
    inj_loss        = yolo_loss(inj_args    ,anchors, num_classes, ignore_thresh=ignore_thresh, print_loss=print_loss)
    
    def true_b():
        return tf.constant(4.0)
    def false_b():
        return tf.constant(2.0)#yolo_loss(golden_args ,anchors, num_classes, ignore_thresh=ignore_thresh, print_loss=print_loss)

    golden_loss = tf.cond(custom_loss_callback.first_run == tf.constant(1),
                          true_b,
                          false_b
    )
    #Combine the loss function
    loss            = custom_loss_combinator(inj_loss,golden_loss)

    return loss


import tensorflow as tf
class CustomLossModel(tf.keras.Model):

    def __init__(self,yolo_body=None,CLASSES=None):
        super(CustomLossModel, self).__init__()
        self.yolo_body = yolo_body
        self.CLASSES   = CLASSES
        self.loss_tracker       = tf.keras.metrics.Mean(name="loss_tot")
        self.loss_tracker_gt    = tf.keras.metrics.Mean(name="loss_inj")
        self.loss_tracker_inj   = tf.keras.metrics.Mean(name="loss_gt")

        
    def set_model(self,model,CLASSES=None):
        self.model      = model
        self.CLASSES    = CLASSES

    # implement the call method
    def call(self, inputs, *args, **kwargs):
       return self.model(inputs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            loss_inj = self(x, training=True)  # Forward pass Injection 
            dx_inj   = tape.gradient(loss_inj, self.trainable_vars)

        with tf.GradientTape() as tape:
            loss_gt  = self(x, training=True)  # Forward pass Injection 
            dx_gt    = tape.gradient(loss_gt, self.trainable_vars)
       
        loss = loss_gt + loss_inj
        
        # Update weights
        self.optimizer.apply_gradients(zip(dx_inj, self.trainable_vars))
        self.optimizer.apply_gradients(zip(dx_gt , self.trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.loss_tracker_inj.update_state(loss_inj)
        self.loss_tracker_gt.update_state(loss_gt)

        #self.mae_metric.update_state(y, y_pred)
        return {"loss_tot": self.loss_tracker.result(),"loss_inj": self.loss_tracker_inj.result(),"loss_gt": self.loss_tracker_gt.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker,self.loss_tracker_inj,self.loss_tracker_gt]
    

    '''
     with tf.GradientTape() as tape:
        loss_inj = self(x, training=True)  # Forward pass Injection 
            
        self.CLASSES.disable_all(verbose=False)         # Disable Classes

        with tf.GradientTape() as tape:   
            loss_gt  = self(x, training=True)  # Forward pass Ground truth
            
            loss = loss_gt + loss_inj           #Sum the loss contribution

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    '''