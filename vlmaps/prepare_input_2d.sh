#!/bin/bash
python -m analysis.input_formatter.input_from_2d_map \
-e "$BASE_DIR/projects/vlmaps-old-version/data/5LpN3gDmAk7_1/map/grid_lseg_1.npy" \
-l "$BASE_DIR/projects/vlmaps-old-version/data/5LpN3gDmAk7_1/map/grid_1_gt.npy" \
-o "$BASE_DIR/projects/vlmaps_python/analysis/input_formatter/map2d.npy" \
