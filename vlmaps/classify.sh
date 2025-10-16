#!/bin/bash
python -m analysis.analysis_tool --dir data/mapdata2/ --batch_out data/mapdata/analysis --classes cfg/mpcat40.tsv -mp -cl --aggregate_classify
python -m analysis.analysis_tool --input data/mapdata/0.data.npy --batch_out data/mapdata/analysis --classes cfg/mpcat40.tsv -mp -cl