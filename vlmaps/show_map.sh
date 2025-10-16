#!/bin/bash
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

cd vlmaps
python -m application.show_map "scene_id=$id"