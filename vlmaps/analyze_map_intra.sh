#!/bin/bash
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

python -m analysis.analysis_tool --input data/gundam-results/$id.data.npy --intra_map -bo test-output
