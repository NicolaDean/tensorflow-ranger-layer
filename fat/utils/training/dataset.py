#Downloading data from Roboflow
from roboflow import Roboflow


def download_dataset(dataset,format="yolokeras"):

    rf = Roboflow(api_key="D5jpG7thd1uxwm3apfHd")
    if   dataset == "aerial":
        project = rf.workspace("jacob-solawetz").project("aerial-maritime")
    elif dataset == "pedestrian":
        project = rf.workspace("roboflow-gw7yv").project("self-driving-car")
    else:
        print("NOT VALID DATASET")
        exit()
    dataset = project.version(3).download("yolokeras")
    print(dataset.location)
 