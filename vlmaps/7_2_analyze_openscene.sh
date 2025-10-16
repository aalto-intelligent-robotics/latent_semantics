#!/bin/bash

# Visual encoder config
if [ -z $1 ]
then
    encoder='lseg'
else
    encoder=$1
fi

if [ -z $2 ]
then
    es=512
else
    es=$2
fi

dir="$OPENSCENE_DIR/parsed-$encoder"

python -m analysis.analysis_tool --dir $dir --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map --cross_dist_analysis --multiprocessing --openscene
