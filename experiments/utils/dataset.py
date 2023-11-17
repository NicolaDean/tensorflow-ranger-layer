
LIBRARY_PATH = "/../"
import sys
import pathlib

# directory reach
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + LIBRARY_PATH)

def load_dataset(DATASET = "MNIST",shape=(32,32)):
  #Load Data from dataset
    if DATASET == "MNIST":
        from our_datasets import mnist
        x_train, y_train, x_val, y_val = mnist.load_data(shape)
        NUM_CLASSES = 10
    elif DATASET == "GTSRB":
        
        from our_datasets import gtsrb
        x_train,x_val,y_train,y_val = gtsrb.load_train(shape)
        #x_test,y_test               = gtsrb.load_test(shape)
        NUM_CLASSES = 43
    elif DATASET == "SHAPE_COUNT":
        from our_datasets import shape_count
        x_train,x_val,y_train,y_val = shape_count.load_dataset(shape)
        NUM_CLASSES = 3
    elif DATASET == "CALTECH101":
        from our_datasets import caltech101
        x_train,x_val,y_train,y_val = caltech101.load_train(shape)
        #x_test,y_test               = caltech101.load_test(shape)
        NUM_CLASSES = 102
    elif DATASET == "STEERING_ANGLE":
        from our_datasets import stearing_angle
        x_train,x_val,y_train,y_val = stearing_angle.load_data(shape)
        NUM_CLASSES = 1
    else:
        print(f"\033[0;31mSELECT A VALID DATASET\033[0m")
        exit()
    return x_train,x_val,y_train,y_val,NUM_CLASSES