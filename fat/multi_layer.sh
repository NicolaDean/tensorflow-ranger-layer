#python single_layer_experiment.py --experiment_name MULTILAYER_MIXED_v3 --frequency 0.5 --mixed_label_v3  --multi_layer --epochs 101
#python single_layer_experiment.py --experiment_name MULTILAYER_MIXED_v3 --frequency 0.75 --mixed_label_v3 --multi_layer --epochs 101

#python single_layer_experiment.py --experiment_name MULTILAYER_B_CL_V2 --frequency 1.0 --custom_loss_v2 --loss_w 1 --multi_layer --epochs 15
#PER IL SECONDO SCRIPT BISOGNA ANDARE A TOCCARE I PESI A MANO (o definirli in un parametro)
#python single_layer_experiment.py --experiment_name MULTILAYER_w_CL_V2 --frequency 1.0 --custom_loss_v2 --loss_w 0.75 --multi_layer --epochs 15

python single_layer_experiment.py --experiment_name PED_MULTILAYER_B_CL_V2 --frequency 1.0 --custom_loss_v2 --loss_w 1  --multi_layer --epochs 6 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5
python single_layer_experiment.py --experiment_name PED_MULTILAYER_W_CL_V2 --frequency 1.0 --custom_loss_v2 --loss_w 0.75 --multi_layer --epochs 6 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5


#TODO
#Rifare il modello CustomLoss Multilayer Boats con classes DEV (o anche il report senza, modello) 
#Fare il Report di PEDESTRIAN LOSS MULTILAYER   (mezza giornata)
#FIXARE golden label per pedestrian 
#FARE modello e report Mixed 50/75 di Pedestrian  (1 giorno)
#REGRESSION

#python single_layer_experiment.py --experiment_name PED_MULTILAYER_MIXED_v1 --frequency 0.75 --mixed_label --multi_layer --epochs 20 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5
#python single_layer_experiment.py --experiment_name PED_MULTILAYER_MIXED_v1 --frequency 0.5 --mixed_label --multi_layer --epochs 20 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5
#python single_layer_experiment.py --experiment_name PED_MULTILAYER_FREQ --frequency 0.5 --multi_layer --epochs 6 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5
#python single_layer_experiment.py --experiment_name PED_MULTILAYER_FREQ --frequency 0.75 --multi_layer --epochs 6 --dataset_path ./Self-Driving-Car-3/ --init_model ./results/pedestrian.h5


#python single_layer_experiment.py --experiment_name 1_CRE_MULTILAYER_FREQ --frequency 0.5 --multi_layer --epochs 120 --extraction_type 2

#python single_layer_experiment.py --experiment_name MULTI_LAYER_FREQ_DEV  --frequency 0.5   --multi_layer --epochs 61 
#python single_layer_experiment.py --experiment_name MULTI_LAYER_FREQ_DEV  --frequency 0.75  --multi_layer  --epochs 61
#python single_layer_experiment.py --experiment_name MULTI_LAYER_MIXED_DEV --frequency 0.5 --mixed_label  --multi_layer  --epochs 61
#python single_layer_experiment.py --experiment_name MULTI_LAYER_MIXED_DEV --frequency 0.75  --mixed_label --multi_layer  --epochs 61
#--extraction_type 2 con 101 epoche

#python single_layer_experiment.py --experiment_name MULTI_LAYER_FREQ_ZERO  --frequency 0.5   --multi_layer --epochs 101 --extraction_type 2
#./report_multi_layer.sh
./report_pre_fat.sh