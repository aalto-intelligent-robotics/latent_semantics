#!/bin/bash
python parse_files.py --input instances-lseg --embeddings embeddings-lseg --output parsed-lseg
python parse_files.py --input instances-openseg --embeddings embeddings-openseg --output parsed-openseg