#!/bin/bash
python instance_classification_benchmark.py --input instances-openseg --output $BASE_DIRcode/openscene/results-mp40 --suffix openseg --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv
#python instance_classification_benchmark.py --input instances-lseg --output $BASE_DIRcode/openscene/results-mp40 --suffix lseg --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv
