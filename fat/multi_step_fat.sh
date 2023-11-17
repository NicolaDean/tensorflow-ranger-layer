
multi_step_fat(){
    #python multi_step_fat.py 
    mkdir ./MULTI_STEP_FAT/$2
    mkdir ./MULTI_STEP_FAT/$2/checkpoints
    mkdir ./MULTI_STEP_FAT/$2/reports
    INIT_MODEL=./../../keras-yolo3/yolo_boats_final.h5
    REPORT_PATH=./reports/yolo/F1_REPORT_BOATS_DEV_PRE_FAT_SINGLE_LAYER.csv
    for i in $(seq 0 $1)
    do
        #ONLY ONE OF THE THREE AT TIME!!
        #FREQ
        #python multi_step_fat.py --experiment_name $2 --init_model $INIT_MODEL  --frequency 0.5 --multi_layer --epochs 5 --msfat $i --starting_point_report $REPORT_PATH 
        #MIXED
        #python multi_step_fat.py --experiment_name $2 --init_model $INIT_MODEL  --frequency 0.5 --mixed_label --multi_layer --epochs 5 --msfat $i --starting_point_report $REPORT_PATH 
        #CUSTOM LOSS
        python multi_step_fat.py --experiment_name $2 --init_model $INIT_MODEL  --frequency 1 --custom_loss_v2 --multi_layer --epochs 5 --msfat $i --starting_point_report $REPORT_PATH 
        #RICORDA DI SOSTITUIRE
        #INIT_MODEL=./MULTI_STEP_FAT/$2/checkpoints/$2_STEP_${i}_0.5.h5
        #REPORT_PATH=./MULTI_STEP_FAT/$2/reports/F1_REPORT_$2_STEP_${i}_0.5.csv
        INIT_MODEL=./MULTI_STEP_FAT/$2/checkpoints/$2_STEP_${i}_1.0.h5
        REPORT_PATH=./MULTI_STEP_FAT/$2/reports/F1_REPORT_$2_STEP_${i}_1.0.csv
        ./msfat_reports.sh $REPORT_PATH $INIT_MODEL 5
    done
    REPORT_PATH=./MULTI_STEP_FAT/$2/reports/F1_final_REPORT.csv
    echo $INIT_MODEL
    ./msfat_reports.sh $REPORT_PATH $INIT_MODEL 50

}

#1: Hardening Step
#2: Experiment Name
#3: 
#This procedure will run a Multi Step FAT composed of three iteretion of the algorithm
#The results will be in ./MULTI_STEP_FAT/$2/reports while the checkpoints in ./MULTI_STEP_FAT/$2/checkpoints
multi_step_fat 14 CUSTOM_LOSS

