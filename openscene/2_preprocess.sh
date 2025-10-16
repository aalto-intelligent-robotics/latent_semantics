#!/bin/bash
cd  scripts/preprocess/
python preprocess_2d_matterport.py
python preprocess_3d_matterport_vlmap_classes.py
cd ../..