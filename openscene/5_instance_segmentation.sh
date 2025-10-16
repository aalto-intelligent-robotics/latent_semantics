#!/bin/bash
python instance_segmentation.py --input save-openseg --output instances-openseg -mp 32 -w
python instance_segmentation.py --input save-lseg --output instances-lseg -mp 32 -w
