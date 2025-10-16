#!/bin/bash
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

cd vlmaps
python -m application.map_classification "scene_id=$id" "type=3"