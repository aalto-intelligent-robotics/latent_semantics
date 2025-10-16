#!/bin/bash
cd scripts/feature_fusion
python matterport_openseg.py \
    --data_dir $BASE_DIRcode/openscene/data/ \
    --output_dir $BASE_DIRcode/openscene/fusedfeatures-openseg \
    --openseg_model $BASE_DIRcode/openscene/downloads/openseg_exported_clip \
    --process_id_range 0,500\
    --split train