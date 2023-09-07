from utils.fat_experiment import *


'''
injection_points = ["conv2d", "batch_normalization"]
injection_points += ["conv2d_"+str(i) for i in range(1, 10)]
injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]
'''

#injection_points = ["conv2d"]
#injection_points = ["batch_normalization"]
#injection_points = ["conv2d_"+str(i) for i in range(1, 10)]
#injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
injection_points += ["batch_normalization_"+str(i) for i in range(5, 10)]
injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]



LAYER = "batch_normalization_5"
injection_points  = [LAYER]


for layer in injection_points:
    EXPERIMENT_NAME   = "GOLDEN_SINGLE_LAYER_" + layer
    FINAL_WEIGHT_NAME = f"single_layer_{layer}_final.h5"
    EPOCHS            = 36

    print(f"Train for layer: {layer}")
    run_fat_experiment(EPOCHS,EXPERIMENT_NAME,FINAL_WEIGHT_NAME,injection_points = injection_points)