#!/bin/bash
./1_prepare_dataset.sh
./create_links.sh
./2_preprocess.sh
./3_fuse_features.sh
./4_eval.sh
./5_instance_segmentation.sh
./6_benchmark.sh