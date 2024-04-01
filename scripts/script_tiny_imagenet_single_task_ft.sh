#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running on gpu: $1"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results_tiny_imagenet"
if [ "$2" != "" ]; then
    RESULTS_DIR=$2
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"


PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name tiny_imagenet_ft_single_task \
               --datasets tiny_imagenet --num_tasks 1 --network resnet18 \
               --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
               --approach finetune --gpu $1 --seed 1993 \
               --pin_memory True --num_workers 8 --nepochs 100 \
               --save_models \
               --lr 0.1 --lr_min 0.0001 --lr_factor 5 --lr_patience 5 --momentum 0.9 --weight_decay 0.0001
