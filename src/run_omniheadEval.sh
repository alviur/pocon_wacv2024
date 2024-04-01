echo "GPU: $1"
echo "CopyOP run: $2"
echo "Port: $3"

RESULT_DIR=" "
mkdir -p $RESULT_DIR


CUDA_VISIBLE_DEVICES=$1 PYTHONPATH='.' python main_incremental.py \
--gpu 0 --exp_name GD_PFR_CopyOP-$2 --datasets cifar100_noTrans \
--network resnet18 --num_tasks 10 --seed 667 --batch_size 512 \
--nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
--num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 \
--lr_patience 20 --lr_min 5e-7 --eval_omni_head --approach evalOmnihead  --dataset2 cifar100 \
--results_path $RESULT_DIR  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048 --port $3 \
--loadTask1 --loadExpert --kd_method L2 \
--pathModelT1  /home/agomezvi/cifar100_noTrans_d2eOPscratch__name-01182023_112342_073538/


