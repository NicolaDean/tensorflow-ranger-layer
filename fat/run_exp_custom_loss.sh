train_all_model(){
    LAYER=$1
    echo $LAYER
    ARGS="--checkpoint CUSTOM_LOSS_0.5_SINGLE_LAYER_$LAYER --experiment_name CUSTOMLOSS --epoch 21 --layer $1  --root ./results/$LAYER"
    echo $ARGS
    python post_fat_report.py $ARGS

}

train_all_model "batch_normalization_3"
train_all_model "batch_normalization_4"
train_all_model "batch_normalization_5"
train_all_model "batch_normalization_6"
train_all_model "batch_normalization_7"
train_all_model "batch_normalization_8"
train_all_model "batch_normalization_9"
train_all_model "batch_normalization_20"
train_all_model "batch_normalization_30"

train_all_model "conv2d_3"
train_all_model "conv2d_4"
train_all_model "conv2d_5"
train_all_model "conv2d_6"
train_all_model "conv2d_7"
train_all_model "conv2d_8"
train_all_model "conv2d_9"