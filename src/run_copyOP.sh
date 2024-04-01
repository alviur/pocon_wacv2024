
echo "GPU: $1"
echo "CopyOP run: $2"
echo "Port: $3"

RESULT_DIR="/data2/users/btwardow/4_tasks_BarlowTwins/FT-$2"
mkdir -p $RESULT_DIR

## Parameters
# ret_nepochs: number or retrospection/adaptation epochs
# adaptSche: 1 for decreasing influence of adaptetion over time (tasks)
# lambdapRet: weight retrospection
# lambdaExp: weight adaptation
# add --getResults to reproduce results

##models @103
#Copy lambdapRet=1
#/data/users/agomezvi/GD_PFR/cifar100_noTrans_toy_GD_PFR_PFR_epochs-10242022_124435_882741
#Copy lambdapRet=0.8
#/data/users/agomezvi/GD_PFR/cifar100_noTrans_toy_GD_PFR_PFR_epochs-10242022_210352_683551
#Copy lambdapRet=2
#/data/users/agomezvi/GD_PFR/cifar100_noTrans_toy_GD_PFR_PFR_epochs-10252022_093445_161366

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH='.' python main_incremental.py \
--gpu 0 --exp_name GD_PFR_CopyOP-$2 --datasets cifar100_noTrans \
--network resnet18 --num_tasks 4 --seed 667 --batch_size 256 \
--nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
--num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 \
--lr_patience 20 --lr_min 5e-7 --eval_omni_head --loadTask1 \
--kd_method GD_PFR --approach copyOP  --task1_nepochs 500 --dataset2 cifar100 \
--results_path $RESULT_DIR  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048 --port $3 \
--loadTask1 --adaptSche 1  --ret_nepochs 500  --lambdapRet 1 --lambdaExp 1 \
--pathModelT1  /data/users/agomezvi/GD_PFR/cifar100_noTrans_toy_GD_PFR_PFR_epochs-10202022_141203_299956/



