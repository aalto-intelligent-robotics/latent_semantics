encoder=vlmaps_lseg
es=512
dir="$VLMAPS_DIR/data/mapdata/$encoder"
python -m analysis.analysis_tool --input $dir/0.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 0
python -m analysis.analysis_tool --input $dir/1.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 1
python -m analysis.analysis_tool --input $dir/2.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 2
python -m analysis.analysis_tool --input $dir/3.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 3
python -m analysis.analysis_tool --input $dir/4.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 4
python -m analysis.analysis_tool --input $dir/5.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 5
python -m analysis.analysis_tool --input $dir/6.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 6
python -m analysis.analysis_tool --input $dir/7.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 7
python -m analysis.analysis_tool --input $dir/8.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 8
python -m analysis.analysis_tool --input $dir/9.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 9

exit 0

encoder=vlmaps_openseg
es=768
dir="$VLMAPS_DIR/data/mapdata/$encoder"
python -m analysis.analysis_tool --input $dir/0.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 0
python -m analysis.analysis_tool --input $dir/1.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 1
python -m analysis.analysis_tool --input $dir/2.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 2
python -m analysis.analysis_tool --input $dir/3.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 3
python -m analysis.analysis_tool --input $dir/4.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 4
python -m analysis.analysis_tool --input $dir/5.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 5
python -m analysis.analysis_tool --input $dir/6.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 6
python -m analysis.analysis_tool --input $dir/7.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 7
python -m analysis.analysis_tool --input $dir/8.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 8
python -m analysis.analysis_tool --input $dir/9.data.npy --batch_out $dir/analysis --embedding_size $es --classes cfg/mpcat40.tsv --intra_map  --multiprocessing --intra_suffix 9