#!/bin/bash

# Visual encoder config
if [ -z $1 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$1
fi

cd vlmaps

for i in 0 1 2 3 4 5 6 7 8 9
do
    echo "$i/9"

    dir="$BASE_DIR/data/vlmaps/data/mapdata/$encoder"

    if [ -e $dir ]
    then
        echo "Saving map to $dir"
    else
        echo "Creating dir $dir"
        mkdir $dir
    fi

    python -m application.parse_map "scene_id=$i" "map_config/visual_encoder=$encoder" "output=$dir"
done
