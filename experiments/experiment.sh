
train(){
    python run_experiment_generic.py --model $1 --dataset $2  --epochs $3 --regression --input_shape $4 --gen_model_statistics --no_report #--no_train --input_shape 100 $4 #--no_train $4
}

train_all(){
    train vgg16 $1 $2 $3
    train vgg19 $1 $2 $3
    train mobilenetv2 $1 $2 $3
    train resnet50 $1 $2 $3
    train efficientnet $1 $2 $3
    train xception $1 $2 $3
    train inceptionv3 $1 $2 $3
    train densenet $1 $2 $3
    train convnettiny $1 $2 $3
}

gen_report(){
    python run_experiment_generic.py --model $1 --dataset $2 --epochs $3 --no_train --input_shape $4 $5 #--no_train $4
}

run_script(){
    #gen_report $1 MNIST 10 48 '--resume_from AAAA --start_at '$2
    #gen_report $1 GTSRB 25 32 '--resume_from AAAA --start_at '$2
    #gen_report $1 CALTECH101 40 64 '--resume_from AAAA --start_at '$2
    #gen_report $1 SHAPE_COUNT 25 64 '--regression --resume_from AAAA --start_at '$2
    gen_report $1 STEERING_ANGLE 25 64 '--regression --resume_from AAAA --start_at '$2
}


#train_all STEERING_ANGLE 100 64


for skip in 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190
do
    run_script dave $skip
    . '''
    run_script vgg16 $skip
    run_script vgg19 $skip
    run_script mobilenetv2 $skip
    run_script resnet50 $skip
    run_script efficientnet $skip
    run_script convnettiny $skip
    run_script densenet $skip
    run_script xception $skip
    run_script nasnet $skip
    '''
done