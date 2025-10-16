#!/bin/bash
# scene_id
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

# Map type: 
# REGULAR = 0
# POSTPROCESSED = 1
# INSTANCES = 2
# PREDICTED = 3
# VLMAP_INSTANCES = 4
# OUR_INSTANCES = 5
# PREDICTED_POSTPROCESSED = 6
if [ -z $2 ]
then
    type='0'
else
    type=$2
fi

# Visual encoder config
if [ -z $3 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$3
fi

cd ../vlmaps
python -m application.instance_segmentation "scene_id=$id" "type=$type" "map_config/visual_encoder=$encoder"