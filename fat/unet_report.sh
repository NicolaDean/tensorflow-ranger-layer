
for idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
do
    python unet_report.py --layer conv2d_$idx
    python unet_report.py --layer batch_normalization_$idx
done

python unet_report.py --layer max_pooling2d
python unet_report.py --layer max_pooling2d_1
python unet_report.py --layer max_pooling2d_2
python unet_report.py --layer max_pooling2d_3