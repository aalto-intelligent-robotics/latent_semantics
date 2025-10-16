#!/bin/bash
cd vlmaps

for i in 1 2 3 4 5 6 7 8 9 10
do
    echo "$i/10"
    python -m application.create_map "scene_id=$id" "map_config/visual_encoder=vlmaps_lseg"
    python -m application.create_map "scene_id=$id" "map_config/visual_encoder=vlmaps_openseg"
done
