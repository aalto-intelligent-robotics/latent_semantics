#!/bin/bash
vlmaps=$VLMAPS_DIR

python -m embeddings_from_images.embeddings_from_images \
--img_dir $vlmaps/data/images \
--embeddings_dir $vlmaps/data/iout \
--weights_path $vlmaps/models/demo_e200.ckpt \
--crop_size 352 \
--base_size 1080