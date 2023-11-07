import pandas as pd
import pandas as pd
from dataclasses import make_dataclass

def summarize_model(MODEL,DATASET):


    FILENAME = f'{MODEL}_{DATASET}_summary.csv'
    FILENAME_NAN = f'../no_ranger_classification_report/{MODEL}_{DATASET}_summary.csv'
    print(FILENAME)

    nan,clip,th = 0,0,0

    try:
        df      = pd.read_csv(FILENAME,sep=";", decimal = ',',header=None)

        clip = df[6].mean()
        th   = df[7].mean()

    except Exception as e:
        print(e)

    try:
        nan_df  = pd.read_csv(FILENAME_NAN,sep=";", decimal = ',',header=None) 
        nan   = nan_df[7].mean()
    except Exception as e:
        print(e)

    return nan,clip,th


MODEL   = "efficientnet"
DATASET = "MNIST"

summarize_model(MODEL,DATASET)

Summary = make_dataclass("Summary", [("model_name",str),("nan_mnist",float), ("clip_mnist",float),("th_mnist",float),("nan_gtsrb",float), ("clip_gtsrb",float),("th_gtsrb",float),("nan_cal",float),("clip_cal",float),("th_cal",float)])


MODELS = ["vgg16","vgg19","resnet50","mobilenetv2","efficientnet","nasnet"]

for model in MODELS:
    mnist_n, mnist_c, mnist_t = summarize_model(model,"MNIST")
    gtsrb_n, gtsrb_c, gtsrb_t = summarize_model(model,"GTSRB")
    calte_n, calte_c, calte_t = summarize_model(model,"CALTECH101")

    rep = Summary(model,mnist_n, mnist_c, mnist_t,gtsrb_n, gtsrb_c, gtsrb_t, calte_n, calte_c, calte_t)
    print(rep)
    report = pd.DataFrame([rep])
    report.to_csv(f"./summary.csv",header=False,mode = 'a', decimal = ',', sep=';',float_format = '%.2f%%')

