#!/bin/bash
if [ -z $1 ]
then
    echo "usage ./gui.sh <file>"
    exit 0
else
    file=$1
fi

python -m analysis.analysis_gui --file $file