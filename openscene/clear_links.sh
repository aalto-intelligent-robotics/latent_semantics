#!/bin/bash

for label in mp21 mp40
do
    rm ~/data/openscene-$label/save-lseg/save-lseg
    rm ~/data/openscene-$label/save-openseg/save-openseg
    rm ~/data/openscene-$label/eval-lseg/eval-lseg
    rm ~/data/openscene-$label/eval-openseg/eval-openseg
    rm ~/data/openscene-$label/fusedfeatures-lseg/fusedfeatures-lseg
    rm ~/data/openscene-$label/fusedfeatures-openseg/fusedfeatures-openseg
    rm ~/data/openscene-$label/instances-lseg/instances-lseg
    rm ~/data/openscene-$label/instances-openseg/instances-openseg
    rm ~/data/openscene-$label/parsed-lseg/parsed-lseg
    rm ~/data/openscene-$label/parsed-openseg/parsed-openseg
    rm ~/data/openscene-$label/embeddings-lseg/embeddings-lseg
    rm ~/data/openscene-$label/embeddings-openseg/embeddings-openseg
done
rm save-lseg
rm save-openseg
rm eval-lseg
rm eval-openseg
rm fusedfeatures-lseg
rm fusedfeatures-openseg
rm instances-lseg
rm instances-openseg
rm parsed-lseg
rm parsed-openseg
rm embeddings-lseg
rm embeddings-openseg