#!/bin/bash

# Visual encoder config
if [ -z $1 ]
then
    encoder='vlmaps_lseg'
else
    encoder=$1
fi

if [ -z $2 ]
then
    es=512
else
    es=$2
fi

dir="$VLMAPS_DIR/data/mapdata/$encoder"

if [ $encoder == "vlmaps_lseg" ]
then
    echo "                                       "
    echo "                                       "
    echo "    ██╗     ███████╗███████╗ ██████╗   "
    echo "    ██║     ██╔════╝██╔════╝██╔════╝   "
    echo "    ██║     ███████╗█████╗  ██║  ███╗  "
    echo "    ██║     ╚════██║██╔══╝  ██║   ██║  "
    echo "    ███████╗███████║███████╗╚██████╔╝  "
    echo "    ╚══════╝╚══════╝╚══════╝ ╚═════╝   "
    echo "                                       "
    echo "                                       "
elif [ $encoder == "vlmaps_openseg" ]
then
    echo "                                                                  "
    echo "                                                                  "
    echo "     ██████╗ ██████╗ ███████╗███╗   ██╗███████╗███████╗ ██████╗   "
    echo "    ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝██╔════╝   "
    echo "    ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗█████╗  ██║  ███╗  "
    echo "    ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║██╔══╝  ██║   ██║  "
    echo "    ╚██████╔╝██║     ███████╗██║ ╚████║███████║███████╗╚██████╔╝  "
    echo "     ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝   "
    echo "                                                                  "
    echo "                                                                  "
fi

echo "                                            "
echo "                                            "
echo "    ██╗███╗   ██╗████████╗██████╗  █████╗   "
echo "    ██║████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗  "
echo "    ██║██╔██╗ ██║   ██║   ██████╔╝███████║  "
echo "    ██║██║╚██╗██║   ██║   ██╔══██╗██╔══██║  "
echo "    ██║██║ ╚████║   ██║   ██║  ██║██║  ██║  "
echo "    ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝  "
echo "                                            "
echo "                                            "

python -m analysis.analysis_tool --dir $dir --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing

echo "                              "
echo "                              "
echo "     ██████╗██████╗  █████╗   "
echo "    ██╔════╝██╔══██╗██╔══██╗  "
echo "    ██║     ██║  ██║███████║  "
echo "    ██║     ██║  ██║██╔══██║  "
echo "    ╚██████╗██████╔╝██║  ██║  "
echo "     ╚═════╝╚═════╝ ╚═╝  ╚═╝  "
echo "                              "
echo "                              "

python -m analysis.analysis_tool --dir $dir --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --cross_dist_analysis  --multiprocessing
