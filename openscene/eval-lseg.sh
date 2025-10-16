#!/bin/bash
# arg 1: experiment directory
dir=eval-lseg

# arg 2: configuration file to be used
#cfg=config/matterport/ours_openseg_pretrained.yaml
cfg=config/matterport/ours_lseg_pretrained.yaml

# arg 3: type: fusion distill ensemble
#type=fusion
#type=distill
type=ensemble

bash run/eval.sh $dir $cfg $type