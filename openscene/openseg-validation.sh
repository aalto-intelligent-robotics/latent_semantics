#!/bin/bash

bash run/eval.sh eval-openseg-fusion config/matterport/ours_openseg_pretrained_fusion.yaml fusion
bash run/eval.sh eval-openseg-distill config/matterport/ours_openseg_pretrained_distill.yaml distill

python instance_segmentation.py --input save-openseg-fusion --output instances-openseg-fusion -mp 32
python instance_segmentation.py --input save-openseg-distill --output instances-openseg-distill -mp 32

python classification_benchmark.py --input save-openseg-fusion --output $BASE_DIRcode/openscene/results-validation --suffix openseg-fusion --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv -mp 32
python classification_benchmark.py --input save-openseg-distill --output $BASE_DIRcode/openscene/results-validation --suffix openseg-distill --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv -mp 32

python instance_classification_benchmark.py --input instances-openseg-fusion --output $BASE_DIRcode/openscene/results-mp40 --suffix openseg-fusion --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv
python instance_classification_benchmark.py --input instances-openseg-distill --output $BASE_DIRcode/openscene/results-mp40 --suffix openseg-distill --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv

python parse_files.py --input instances-openseg-fusion --embeddings embeddings-openseg-fusion --output parsed-openseg-fusion
python parse_files.py --input instances-openseg-distill --embeddings embeddings-openseg-distill --output parsed-openseg-distill