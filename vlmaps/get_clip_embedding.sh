#!/bin/bash
if [ -z $1 ]
then
    type=text
else
    type=$1
fi

if [ -z $2 ]
then
    echo "please input query. Use: ./get_clip_embedding.sh <text/image> <query/path>"
    exit 0
else
    query=$2
fi

python -m embeddings.get_embeddings --network clip --model_name ViT-B/32 --$1 $2