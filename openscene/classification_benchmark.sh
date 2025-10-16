#!/bin/bash
python classification_benchmark.py --input save-openseg --output $BASE_DIRcode/openscene/results-mp40 --suffix openseg --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv -mp 32
#python classification_benchmark.py --input save-lseg --output $BASE_DIRcode/openscene/results-mp40 --suffix lseg --classes $BASE_DIRcode/vlmaps/cfg/mpcat21.tsv -mp 32
