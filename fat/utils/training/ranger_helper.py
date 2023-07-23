
from tqdm import tqdm

def ranger_domain_tuning(RANGER,train_gen,num_of_batch):  
    #RAGE TUNE THE YOLO MODEL
    print("=============RANGE TUNING=============")
    for _ in tqdm(range(num_of_batch)):
        dataset = next(train_gen)
        data   = dataset[0][0]
        image_data = data
        #image_data = np.expand_dims(data[0], 0)  # Add batch dimension.
        RANGER.tune_model_range(image_data, reset=False)