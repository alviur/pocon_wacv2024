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


if [ "$1" = "ft-base" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name baseline_base \
               --datasets cifar100_icarl --num_tasks 10 --network resnet32 \
               --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach finetune --gpu $2 --all_outputs --stop_at_task 3
elif [ "$1" = "ft-main" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name baseline_main \
               --datasets cifar100_icarl --num_tasks 10 --network resnet32 \
               --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach finetune --gpu $2 --stop_at_task 3
elif [ "$1" = "ft-exem" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name baseline_exem \
               --datasets cifar100_icarl --num_tasks 10 --network resnet32 \
               --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach finetune --gpu $2 \
               --num_exemplars 2000 --exemplar_selection herding
elif [ "$1" = "fz-base" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name baseline_base \
               --datasets cifar100_icarl --num_tasks 10 --network resnet32 \
               --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach freezing --gpu $2 --all_outputs
elif [ "$1" = "fz-main" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name baseline_main \
               --datasets cifar100_icarl --num_tasks 10 --network resnet32 \
               --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach freezing --gpu $2
elif [ "$1" = "fz-exem" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name baseline_exem \
               --datasets cifar100_icarl --num_tasks 10 --network resnet32 \
               --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
               --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
               --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
               --approach freezing --gpu $2 \
               --num_exemplars 2000 --exemplar_selection herding
fi
