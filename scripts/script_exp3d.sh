#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running approach: $1"
else
    echo "No approach has been assigned."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
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


if [ "$1" = "joint" ] || [ "$1" = "dmc" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name exp3d \
               --datasets flowers scenes birds cars aircraft actions --num_tasks 1 --network alexnet \
               --nepochs 200 --batch_size 32 --results_path $RESULTS_DIR \
               --gridsearch_tasks 3 --gridsearch_config gridsearch_config_pretrained \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2 --pretrained
elif [ "$1" = "freezing" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name exp3d \
               --datasets flowers scenes birds cars aircraft actions --num_tasks 1 --network alexnet \
               --nepochs 200 --batch_size 32 --results_path $RESULTS_DIR \
               --gridsearch_tasks 3 --gridsearch_config gridsearch_config_pretrained \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2 --pretrained --freeze_at -1
elif [ "$1" = "lwm" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name exp3d \
               --datasets flowers scenes birds cars aircraft actions --num_tasks 1 --network alexnet \
               --nepochs 200 --batch_size 32 --results_path $RESULTS_DIR \
               --gridsearch_tasks 3 --gridsearch_config gridsearch_config_pretrained \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2 --pretrained --gradcam_layer features
else
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name exp3d \
               --datasets flowers scenes birds cars aircraft actions --num_tasks 1 --network alexnet \
               --nepochs 200 --batch_size 32 --results_path $RESULTS_DIR \
               --gridsearch_tasks 3 --gridsearch_config gridsearch_config_pretrained \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2  --pretrained \
               --num_exemplars_per_class 5 --exemplar_selection herding
fi
