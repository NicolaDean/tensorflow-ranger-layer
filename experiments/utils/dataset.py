
LIBRARY_PATH = "/../"
import sys
import pathlib

# directory reach
directory = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(directory + LIBRARY_PATH)

def load_dataset(DATASET = "MNIST"):
  #Load Data from dataset
    if DATASET == "MNIST":
        from datasets import mnist
        x_train, y_train, x_val, y_val = mnist.load_data()
        NUM_CLASSES = 10
    elif DATASET == "GTSRB":
        
        from datasets import gtsrb
        x_train,x_val,y_train,y_val = gtsrb.load_train()
        x_test,y_test               = gtsrb.load_test()
        NUM_CLASSES = 43
    else:
        print(f"\033[0;31mSELECT A VALID DATASET\033[0m")
        exit()
    return x_train,x_val,y_train,y_val,NUM_CLASSES