#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running on gpu: $1"
    GPU="$1"
else
    echo "No gpu has been assigned."
fi

if [ "$2" != "" ]; then
    echo "NUM_TASKS: $2"
    NUM_TASKS="$2"
else
    echo "You need to type number of tasks!."
    exit
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results_emb"
if [ "$3" != "" ]; then
    RESULTS_DIR=$3
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

SEED=7
if [ "$4" != "" ]; then
    SEED=$4
else
    echo "No seed is given"
fi
echo "SEED: $SEED"


PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name align_triplet_${NUM_TASKS}_seed_${SEED} \
       --datasets cifar10 --num_tasks $NUM_TASKS --network resnet32 \
       --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
       --lr 0.01 --momentum 0.9 --weight_decay 5e-4 --lr_patience 15 --lr_factor 10 --lr_min 0.000001 \
       --approach emb_align --gpu $GPU --seed $SEED \
       --fix_bn --metric_loss align_uniform
