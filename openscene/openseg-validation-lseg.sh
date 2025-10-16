#!/bin/bash

bash run/eval.sh eval-lseg-fusion config/matterport/ours_lseg_pretrained_fusion.yaml fusion
bash run/eval.sh eval-lseg-distill config/matterport/ours_lseg_pretrained_distill.yaml distill

python instance_segmentation.py --input save-lseg-fusion --output instances-lseg-fusion -mp 32
python instance_segmentation.py --input save-lseg-distill --output instances-lseg-distill -mp 32

python classification_benchmark.py --input save-lseg-fusion --output $BASE_DIRcode/openscene/results-validation --suffix lseg-fusion --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv -mp 32
python classification_benchmark.py --input save-lseg-distill --output $BASE_DIRcode/openscene/results-validation --suffix lseg-distill --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv -mp 32

python instance_classification_benchmark.py --input instances-lseg-fusion --output $BASE_DIRcode/openscene/results-mp40 --suffix lseg-fusion --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv
python instance_classification_benchmark.py --input instances-lseg-distill --output $BASE_DIRcode/openscene/results-mp40 --suffix lseg-distill --classes $BASE_DIRcode/vlmaps/cfg/mpcat40.tsv

python parse_files.py --input instances-lseg-fusion --embeddings embeddings-lseg-fusion --output parsed-lseg-fusion
python parse_files.py --input instances-lseg-distill --embeddings embeddings-lseg-distill --output parsed-lseg-distill