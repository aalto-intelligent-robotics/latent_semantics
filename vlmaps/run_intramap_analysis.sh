#!/bin/bash

./6_parse_maps.sh vlmaps_lseg
./6_parse_maps.sh vlmaps_openseg
./7_analyze_maps.sh vlmaps_lseg 512 --intra_map
./7_analyze_maps.sh vlmaps_openseg 768 '--cross_dist_analysis --intra_map'
