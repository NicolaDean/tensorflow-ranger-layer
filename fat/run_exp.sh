#!/bin/bash
# Read a string with spaces using for loop

conda activate nicola
for value in 1 2 3 4 5 6 7 8 9 25 42 56
do
   for epoch in 5 10 15 20 25
   do
      python post_fat_report.py --layer conv2d_$value --epoch $epoch
   done
done

for value in 1 2 3 4 5 6 7 8 9 25 42 56
do
   for epoch in 5 10 15 20 25
   do
      python post_fat_report.py --layer batch_normalization_$value --epoch $epoch 
   done
done
