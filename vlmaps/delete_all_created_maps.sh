#!/bin/bash
#!/bin/bash

function yes_or_no {
    while true; do
        read -p "$* [y/n]: " yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) echo "Aborted" ; return  1 ;;
        esac
    done
}

function go_for_it {
    rm -rf $DATA_DIR/vlmaps_dataset/5LpN3gDmAk7_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/gTV8FGcVJC9_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/jh4fc5c5qoQ_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/JmbYfDe2QKZ_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/JmbYfDe2QKZ_2/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/mJXqzFtmKg4_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/ur6pFq6Qu1A_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/UwV83HsGsw3_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/Vt2qJdWjCF2_1/vlmap
    rm -rf $DATA_DIR/vlmaps_dataset/YmJkqBEsHnH_1/vlmap
}

message="This will delete all VLMaps. Are you sure?"

yes_or_no "$message" && go_for_it