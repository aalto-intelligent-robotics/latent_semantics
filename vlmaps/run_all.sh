#!/bin/bash
# data directory
if [ -z $1 ]
then
    dir='0'
else
    dir=$1
fi

##################################################
# DATA: Downloading, Generation and Preprocessing
##################################################

cd scripts

# Download all Matterport scenes processed by VLMaps
if [ -e $dir/matterport3d/v1/ ]
then
    echo "Data folder exists. Skip downloading."
else
    echo "Downloading data..."
    ./downloader.sh $dir/matterport3d/
fi

# Unzip data
if [ -e $dir/matterport3d/v1/tasks/mp3d/ ]
then
    echo "mp3d exists."
else
    echo "Unzipping mp3d..."
    unzip $dir/matterport3d/v1/tasks/mp3d_habitat.zip -d $dir/matterport3d/v1/tasks/
fi

# Generate dataset
./generate_dataset.sh

# Check dataset
./check_initial_dataset.sh

##################################################
# MAPS: Map creation, Postprocessing, Analysis
##################################################

for encoder in vlmaps_lseg vlmaps_openseg
do
    echo "================================="
    echo "Encoder: $encoder"
    echo "================================="

    for i in 0 1 2 3 4 5 6 7 8 9
    do
        echo "$i/10"
        ./run_one_scene.sh $i $encoder
    done
done
