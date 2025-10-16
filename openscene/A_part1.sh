#!/bin/bash
./1_prepare_dataset.sh
./create_links.sh
./2_preprocess.sh
./3_fuse_features.sh