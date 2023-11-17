
unet_rep(){
    for idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
    do
        python $1 --layer conv2d_$idx
        python $1 --layer batch_normalization_$idx
    done

    python $1 --layer max_pooling2d
    python $1 --layer max_pooling2d_1
    python $1 --layer max_pooling2d_2
    python $1 --layer max_pooling2d_3
}

unet_rep unet_report_pet.py