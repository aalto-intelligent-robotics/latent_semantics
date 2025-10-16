#!/bin/bash
python -m image_gt_generator.coco --num 3 --out_path data/iout_small/image_data --subsample 100 --cfg_path cfg/image_cats.cfg --search image_gt_generator/small.cfg --split