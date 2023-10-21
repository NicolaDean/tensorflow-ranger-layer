
gen_report(){
    python run_experiment_generic.py --model $1 --dataset $2 --epochs $3 --no_train $4
}

run_script(){
    #gen_report $1 MNIST 10 '--resume_from AAAA --start_at '$2
    gen_report $1 GTSRB 25 '--resume_from AAAA --start_at '$2
}

run_script vgg19 0


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
for skip in 200 220 240 260 280 300  
do
    #run_script mobilenetv2 $skip
    #run_script resnet50 $skip
    #run_script efficientnet $skip
    #run_script convnettiny $skip
    #run_script densenet $skip
done