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

cd ../vlmaps
python -m application.vlmap_instance_segmentation "scene_id=$id" "map_config/visual_encoder=$encoder"