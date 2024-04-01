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

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$3" != "" ]; then
    RESULTS_DIR=$3
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

NETWORK=$4

if [ "$1" = "joint" ] || [ "$1" = "dmc" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name subset_imagenet_${NETWORK} \
               --datasets imagenet_subset --num_tasks 10 --network $NETWORK \
               --nepochs 100 --batch_size 128 --results_path $RESULTS_DIR \
               --momentum 0.9 --weight_decay 0.0002 --lr 0.1 --lr_patience 10 \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config_subset \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2
else
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name subset_imagenet_${NETWORK} \
               --datasets imagenet_subset --num_tasks 10 --network $NETWORK \
               --nepochs 100 --batch_size 128 --results_path $RESULTS_DIR \
               --momentum 0.9 --weight_decay 0.0002 --lr 0.1 --lr_patience 10 \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config_subset \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2 \
               --num_exemplars_per_class 20 --exemplar_selection herding
fi
