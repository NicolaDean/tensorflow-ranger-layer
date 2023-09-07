import os 
from os import listdir
from os.path import isfile, join


injection_points = ["conv2d", "batch_normalization"]
injection_points += ["conv2d_"+str(i) for i in range(1, 10)]
injection_points += ["batch_normalization_"+str(i) for i in range(2, 10)]
injection_points += ["conv2d_25","conv2d_42","conv2d_56","conv2d_71"]
injection_points += ["batch_normalization_25", "batch_normalization_42", "batch_normalization_56", "batch_normalization_71"]


for layer in injection_points:
    try:
        name = f'./SINGLE_LAYER_{layer}'
        dir_list = os.listdir(name)
        #print(dir_list)
        os.chdir(name)
        for n in dir_list:
            new_name = n.split('-')[0] + "-" + n.split('-')[1] + ".h5"
            print(new_name)
            os.rename(n,new_name)
        os.chdir('./..')
    except:
        print(f"{layer} not exist")