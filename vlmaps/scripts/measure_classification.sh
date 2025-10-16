#!/bin/bash
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

if [ -z $2 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$2
fi

cd ../vlmaps
python -m application.measure_classification "scene_id=$id" "map_config/visual_encoder=$encoder"