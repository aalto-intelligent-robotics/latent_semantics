#!/bin/bash
if [ -z $1 ]
then
    type=0
else
    type=$1
fi

if [ $type -eq 0 ]
then
    python -m analysis.analysis_gui --file data/mapdata/analysis/batch.out
else
    python -m analysis.analysis_gui --file data/mapdata/analysis/stats.out
fi
