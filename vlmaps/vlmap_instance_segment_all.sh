#!/bin/bash
# Map prefix
if [ -z $1 ]
then
    prefix='vlmaps'
else
    prefix=$1
fi

# PREDICTED(3)  --> [vlmap_instance_segmentation]   --> VLMAP_INSTANCES(4)

echo "1/10"
./vlmap_instance_segmentation.sh 0 $prefix
echo "2/10"
./vlmap_instance_segmentation.sh 1 $prefix
echo "3/10"
./vlmap_instance_segmentation.sh 2 $prefix
echo "4/10"
./vlmap_instance_segmentation.sh 3 $prefix
echo "5/10"
./vlmap_instance_segmentation.sh 4 $prefix
echo "6/10"
./vlmap_instance_segmentation.sh 5 $prefix
echo "7/10"
./vlmap_instance_segmentation.sh 6 $prefix
echo "8/10"
./vlmap_instance_segmentation.sh 7 $prefix
echo "9/10"
./vlmap_instance_segmentation.sh 8 $prefix
echo "10/10"
./vlmap_instance_segmentation.sh 9 $prefix