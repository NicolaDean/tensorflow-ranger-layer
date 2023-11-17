train_all_model(){
    $LAYER=$1
    ARGS="--experiment_name CUSTOM_LOSS --layer $1 --custom_loss_v2 --epochs 21"
    python single_layer_experiment.py $ARGS
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
