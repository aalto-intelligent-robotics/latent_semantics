#!/bin/bash
# scene_id
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

# Visual encoder config
if [ -z $2 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$2
fi


# Map type: 
# * REGULAR = 0
# * POSTPROCESSED = 1
# * INSTANCES = 2
# * PREDICTED = 3
# * VLMAP_INSTANCES = 4
# * OUR_INSTANCES = 5
# * PREDICTED_POSTPROCESSED = 6

# Processing of Ground-truth instances
# REGULAR(0)                  --> [semantic_postprocessing]       --> POSTPROCESSED(1)
# POSTPROCESSED(1)            --> [instance_segmentation]         --> INSTANCES(2)

# Postprocessing of map with embeddings
# REGULAR(0)                  --> [map_classification]            --> PREDICTED(3)
# PREDICTED(3)                --> [semantic_postprocessing]       --> PREDICTED_POSTPROCESSED(6)
# PREDICTED_POSTPROCESSED(6)  --> [instance_segmentation]         --> OUR_INSTANCES(5)
# PREDICTED(3)                --> [vlmap_instance_segmentation]   --> VLMAP_INSTANCES(4)

# Create the base map
./create_map.sh $id $encoder

# Create GT maps
./semantic_postprocessing.sh $id 0 $encoder
./instance_segmentation.sh $id 1 $encoder

# Postprocessing
./predict_map.sh $id 0 $encoder
./semantic_postprocessing.sh $id 3 $encoder
./instance_segmentation.sh $id 6 $encoder
./vlmap_instance_segmentation.sh $id $encoder

# Analysis
./measure_classification.sh $id $encoder
./measure_instance_classification.sh $id $encoder
