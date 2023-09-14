import sys
import pathlib

directory = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(directory +  "/../../../keras-yolo3")


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
    yolo_outputs    = args[:num_layers]

    #GET THE LABELS
    vanilla_out     = args[num_layers:num_layers*2]
    golden_out      = args[num_layers*2:]

    #Define Loss args
    vanilla_args    = [*yolo_outputs, *vanilla_out]
    golden_args     = [*yolo_outputs, *golden_out ]

    #Compute the "local loss"
    vanilla_loss    = yolo_loss(vanilla_args,anchors, num_classes, ignore_thresh=ignore_thresh, print_loss=print_loss)
    golden_loss     = yolo_loss(golden_args ,anchors, num_classes, ignore_thresh=ignore_thresh, print_loss=print_loss)
    
    #Combine the loss function
    loss            = custom_loss_combinator(vanilla_loss,golden_loss)

    return loss
        