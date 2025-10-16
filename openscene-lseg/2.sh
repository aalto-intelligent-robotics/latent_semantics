data=$BASE_DIRcode/openscene/data
out=$BASE_DIRhdd/openscene-mp40/fusedfeatures-lseg
model=$BASE_DIRcode/openscene-lseg/checkpoints/demo_e200.ckpt

python fusion_matterport.py --data_dir $data --output_dir $out --lseg_model $model --process_id_range 0,500 --split train