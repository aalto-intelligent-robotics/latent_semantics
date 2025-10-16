#!/bin/bash

# Visual encoder config
if [ -z $1 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$1
fi

# Embedding size
if [ -z $2 ]
then
    es=512
else
    es=$2
fi

# Additional args for the analyzer, such as --intra_map --cross_dist_analysis
if [ -z $3 ]
then
    aargs=''
else
    aargs=$3
fi

dir="$BASE_DIR/data/vlmaps/data/mapdata/$encoder"

python -m analysis.analysis_tool --dir $dir --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --multiprocessing $aargs
