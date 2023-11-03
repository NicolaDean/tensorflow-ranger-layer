#python single_layer_experiment.py --experiment_name MULTILAYER_MIXED_v3 --frequency 0.5 --mixed_label_v3  --multi_layer --epochs 101
#python single_layer_experiment.py --experiment_name MULTILAYER_MIXED_v3 --frequency 0.75 --mixed_label_v3 --multi_layer --epochs 101

#python single_layer_experiment.py --experiment_name PED_MULTILAYER_MIXED_v1 --frequency 0.75 --mixed_label --multi_layer --epochs 20 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5
#python single_layer_experiment.py --experiment_name PED_MULTILAYER_MIXED_v1 --frequency 0.5 --mixed_label --multi_layer --epochs 20 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5
#python single_layer_experiment.py --experiment_name PED_MULTILAYER_FREQ --frequency 0.5 --multi_layer --epochs 20 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5
#python single_layer_experiment.py --experiment_name PED_MULTILAYER_FREQ --frequency 0.75 --multi_layer --epochs 20 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5


#python single_layer_experiment.py --experiment_name 1_CRE_MULTILAYER_FREQ --frequency 0.5 --multi_layer --epochs 120 --extraction_type 2

#python single_layer_experiment.py --experiment_name MULTI_LAYER_FREQ_DEV  --frequency 0.5   --multi_layer --epochs 61 
#python single_layer_experiment.py --experiment_name MULTI_LAYER_FREQ_DEV  --frequency 0.75  --multi_layer  --epochs 61
#python single_layer_experiment.py --experiment_name MULTI_LAYER_MIXED_DEV --frequency 0.5 --mixed_label  --multi_layer  --epochs 61
#python single_layer_experiment.py --experiment_name MULTI_LAYER_MIXED_DEV --frequency 0.75  --mixed_label --multi_layer  --epochs 61
#--extraction_type 2 con 101 epoche

#python single_layer_experiment.py --experiment_name MULTI_LAYER_FREQ_ZERO  --frequency 0.5   --multi_layer --epochs 101 --extraction_type 2
./report_multi_layer.sh