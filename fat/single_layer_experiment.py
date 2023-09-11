from utils.fat_experiment import *
import sys
import argparse
import os
from tqdm import tqdm

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


for freq in [1.0,0.75,0.5,0.25]:
    for layer in injection_points:
        EXPERIMENT_NAME   = f"FREQUENCY_{freq}__SINGLE_LAYER_" + layer
        FINAL_WEIGHT_NAME = f"single_layer_{layer}_final.h5"
        EPOCHS            = 36
        INJECTION_FREQUENCY = freq
        GOLDEN_LABEL      = False

        print(f"Train for layer: {layer}")
        run_fat_experiment(EPOCHS,EXPERIMENT_NAME,FINAL_WEIGHT_NAME,injection_points = injection_points,GOLDEN_LABEL=GOLDEN_LABEL,injection_frequency=INJECTION_FREQUENCY)

for freq in [1.0,0.75,0.5,0.25]:
    for layer in injection_points:
        EXPERIMENT_NAME   = f"GOLDEN_FREQUENCY_{freq}__SINGLE_LAYER_" + layer
        FINAL_WEIGHT_NAME = f"single_layer_{layer}_final.h5"
        EPOCHS            = 36
        INJECTION_FREQUENCY = freq
        GOLDEN_LABEL      = True

        print(f"Train for layer: {layer}")
        run_fat_experiment(EPOCHS,EXPERIMENT_NAME,FINAL_WEIGHT_NAME,injection_points = injection_points,GOLDEN_LABEL=GOLDEN_LABEL,injection_frequency=INJECTION_FREQUENCY)


'''
for layer in injection_points:
    EXPERIMENT_NAME   = "FREQUENCY_0.75__SINGLE_LAYER_" + layer
    FINAL_WEIGHT_NAME = f"single_layer_{layer}_final.h5"
    EPOCHS            = 36
    INJECTION_FREQUENCY = 0.75
    GOLDEN_LABEL      = False

    print(f"Train for layer: {layer}")
    run_fat_experiment(EPOCHS,EXPERIMENT_NAME,FINAL_WEIGHT_NAME,injection_points = injection_points,GOLDEN_LABEL=GOLDEN_LABEL,injection_frequency=INJECTION_FREQUENCY)
'''
