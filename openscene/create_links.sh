#!/bin/bash
if [ -z $1 ]
then
    labels=mp21
else
    labels=$1
fi

# data dir
mkdir -p ~/data/openscene-data
ln -s ~/data/openscene-data data

# evaluation dirs
mkdir -p ~/data/openscene-$labels/save-lseg
mkdir -p ~/data/openscene-$labels/save-openseg
mkdir -p ~/data/openscene-$labels/eval-lseg
mkdir -p ~/data/openscene-$labels/eval-openseg
mkdir -p ~/data/openscene-$labels/fusedfeatures-lseg
mkdir -p ~/data/openscene-$labels/fusedfeatures-openseg
mkdir -p ~/data/openscene-$labels/instances-lseg
mkdir -p ~/data/openscene-$labels/instances-openseg
mkdir -p ~/data/openscene-$labels/parsed-lseg
mkdir -p ~/data/openscene-$labels/parsed-openseg
mkdir -p ~/data/openscene-$labels/embeddings-lseg
mkdir -p ~/data/openscene-$labels/embeddings-openseg

ln -s ~/data/openscene-$labels/save-lseg save-lseg
ln -s ~/data/openscene-$labels/save-openseg save-openseg
ln -s ~/data/openscene-$labels/eval-lseg eval-lseg
ln -s ~/data/openscene-$labels/eval-openseg eval-openseg
ln -s ~/data/openscene-$labels/fusedfeatures-lseg fusedfeatures-lseg
ln -s ~/data/openscene-$labels/fusedfeatures-openseg fusedfeatures-openseg
ln -s ~/data/openscene-$labels/instances-lseg instances-lseg
ln -s ~/data/openscene-$labels/instances-openseg instances-openseg
ln -s ~/data/openscene-$labels/parsed-lseg parsed-lseg
ln -s ~/data/openscene-$labels/parsed-openseg parsed-openseg
ln -s ~/data/openscene-$labels/embeddings-lseg embeddings-lseg
ln -s ~/data/openscene-$labels/embeddings-openseg embeddings-openseg
ln -s ../vlmaps/vlmaps/utils utils

