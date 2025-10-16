#!/bin/bash
# Map type: 
# REGULAR = 0
# POSTPROCESSED = 1
# INSTANCES = 2
# PREDICTED = 3
# VLMAP_INSTANCES = 4
# OUR_INSTANCES = 5
# PREDICTED_POSTPROCESSED = 6
if [ -z $1 ]
then
    type='0'
else
    type=$1
fi

# Map prefix
if [ -z $2 ]
then
    prefix='vlmaps'
else
    prefix=$2
fi

./semantic_postprocessing.sh 0 $type $prefix
./semantic_postprocessing.sh 1 $type $prefix
./semantic_postprocessing.sh 2 $type $prefix
./semantic_postprocessing.sh 3 $type $prefix
./semantic_postprocessing.sh 4 $type $prefix
./semantic_postprocessing.sh 5 $type $prefix
./semantic_postprocessing.sh 6 $type $prefix
./semantic_postprocessing.sh 7 $type $prefix
./semantic_postprocessing.sh 8 $type $prefix
./semantic_postprocessing.sh 9 $type $prefix