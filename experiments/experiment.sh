
train(){
    python run_experiment_generic.py --model $1 --dataset SHAPE_COUNT --epochs 25 --regression --input_shape 100 --gen_model_statistics --no_report #--no_train --input_shape 100 $4 #--no_train $4
}

gen_report(){
    python run_experiment_generic.py --model $1 --dataset $2 --epochs $3 --no_train --input_shape $4 $5 #--no_train $4
}

run_script(){
    #gen_report $1 MNIST 10 48 '--resume_from AAAA --start_at '$2
    #gen_report $1 GTSRB 25 32 '--resume_from AAAA --start_at '$2
    #gen_report $1 CALTECH101 40 64 '--resume_from AAAA --start_at '$2
    #gen_report $1 SHAPE_COUNT 25 100 '--regression --resume_from AAAA --start_at '$2
    gen_report $1 STEERING_ANGLE 25 64 '--regression --resume_from AAAA --start_at '$2
}


#train vgg16 0
#train vgg19 0
#train mobilenetv2 0
#train resnet50 0
#train efficientnet 0
#train xception 0
#train inceptionv3 0
#train densenet 0
#train convnettiny 0


#run_script vgg16
#run_script vgg19 
#run_script mobilenetv2 
#run_script resnet50
#run_script efficientnet 
#run_script convnettiny
#run_script densenet
#run_script inceptionv3  #NEED RESIZE (75x75)
#run_script nasnet    #NEED RESIZE
#run_script xception  #NEED RESIZE

#gen_report mobilenetv2 MNIST 5 '--resume_from block_1_expand_BN'
#gen_report mobilenetv2 GTSRB 25 '--resume_from block_2_depthwise_BN'
for skip in 1 #5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190
do
    run_script dave $skip
    #run_script vgg16 $skip
    #run_script vgg19 $skip
    #run_script mobilenetv2 $skip
    #run_script resnet50 $skip
    #run_script efficientnet $skip
    #run_script convnettiny $skip
    #run_script densenet $skip
    #run_script xception $skip
    #run_script nasnet $skip

done