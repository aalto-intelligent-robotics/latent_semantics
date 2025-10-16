#!/bin/bash
if [ -z $1 ]
then
    prefix='vlmaps'
else
    prefix=$1
fi

./measure_instance_classification.sh 0 $prefix
./measure_instance_classification.sh 1 $prefix
./measure_instance_classification.sh 2 $prefix
./measure_instance_classification.sh 3 $prefix
./measure_instance_classification.sh 4 $prefix
./measure_instance_classification.sh 5 $prefix
./measure_instance_classification.sh 6 $prefix
./measure_instance_classification.sh 7 $prefix
./measure_instance_classification.sh 8 $prefix
./measure_instance_classification.sh 9 $prefix