#!/bin/bash
python -m analysis.analysis_tool --dir data/iout --batch_out data/iout_small/analysis --classes cfg/image_cats.cfg --delimiter ";" -mp --debug  --batch -cda
python -m analysis.analysis_tool --dir data/iout_small --batch_out data/iout/analysis --classes cfg/small.cfg --delimiter ";" -mp --batch -cda