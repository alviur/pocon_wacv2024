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

RESULTS_DIR="$PROJECT_DIR/results_exemplars"
if [ "$2" != "" ]; then
    RESULTS_DIR=$2
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"


for EXEMPLARS_PER_CLASS in 0 20 10 30 50
do
PYTHONPATH=$SRC_DIR python -u $SRC_DIR/main_incremental.py --exp_name cifar100_10tasks_kd_only_new_ce_multihead \
       --datasets cifar100_icarl --num_tasks 10 --network resnet32 \
       --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
       --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
       --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
       --approach lwf --gpu $1 --kd_only_new --ce_exemplars_task_aware \
       --num_exemplars_per_class $EXEMPLARS_PER_CLASS --exemplar_selection herding
done
