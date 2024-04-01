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
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name exp4b \
               --datasets exp4_1-1 exp4_1-2 exp4_1-3 exp4_1-4 \
                          exp4_2-1 exp4_2-2 exp4_2-3 exp4_2-4 \
                          exp4_3-1 exp4_3-2 exp4_3-3 exp4_3-4 \
                          exp4_4-1 exp4_4-2 exp4_4-3 exp4_4-4 --num_tasks 1 --network alexnet \
               --nepochs 200 --batch_size 32 --results_path $RESULTS_DIR \
               --gridsearch_tasks 16 --gridsearch_config gridsearch_config_pretrained \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2
else
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name exp4b \
               --datasets exp4_1-1 exp4_1-2 exp4_1-3 exp4_1-4 \
                          exp4_2-1 exp4_2-2 exp4_2-3 exp4_2-4 \
                          exp4_3-1 exp4_3-2 exp4_3-3 exp4_3-4 \
                          exp4_4-1 exp4_4-2 exp4_4-3 exp4_4-4 --num_tasks 1 --network alexnet \
               --nepochs 200 --batch_size 32 --results_path $RESULTS_DIR \
               --gridsearch_tasks 16 --gridsearch_config gridsearch_config_pretrained \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach $1 --gpu $2 --gridsearch_max_num_searches 15 \
               --num_exemplars_per_class 2 --exemplar_selection herding
fi
