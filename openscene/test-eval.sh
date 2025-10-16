#!/bin/bash
dir=eval-openseg

# arg 2: configuration file to be used
cfg=config/matterport/ours_openseg_pretrained.yaml


bash run/eval.sh $dir $cfg distill