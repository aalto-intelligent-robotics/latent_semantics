#!/bin/bash
if [ -z $1 ]
then
    id='0'
else
    id=$1
fi

python -m analysis.analysis_tool --dir data/gundam-results/ -cda -bo data/gundam-results/test-out
