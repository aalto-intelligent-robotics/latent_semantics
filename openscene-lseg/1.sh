for scene in 5LpN3gDmAk7_1 gTV8FGcVJC9_1 jh4fc5c5qoQ_1 JmbYfDe2QKZ_1 JmbYfDe2QKZ_2 mJXqzFtmKg4_1 ur6pFq6Qu1A_1 UwV83HsGsw3_1 Vt2qJdWjCF2_1 YmJkqBEsHnH_1
do
    echo "running scene: $scene"
    # arg 1: RBG images
    rgb=$BASE_DIRhdd/datasets/vlmaps_dataset/$scene/rbg

    # arg 2: embeddings output
    out=$BASE_DIRhdd/datasets/vlmaps_dataset/$scene/embeddings

    # arg 3: width
    width=1080

    python lseg_feature_extraction.py --data_dir $rgb --output_dir $out --img_long_side $width
done