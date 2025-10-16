#!/bin/bash

# Visual encoder config
if [ -z $1 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$1
fi

for i in 0 1 2 3 4 5 6 7 8 9
do
    echo "$i/10"
    ./measure_classification.sh $i $encoder
    ./measure_instance_classification.sh $i $encoder
done
