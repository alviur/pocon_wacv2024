from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 9 --exp_name POCON_CIFAR100 --eval_omni_head  --nepochs 1 --ret_nepochs 1 --task1_Retnepochs 1 --head_classifier_lr 5e-2 --classifier_nepochs 1 --lambdapRet 25 -lr_factor 3 --lr_patience 20 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 10 --seed 667 --batch_size 256  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5  --lr_min 5e-7   --kd_method L2  --approach copyOPscratch --projectorArc 2048_2048_2048 --wandblog   --expertArch ResNet18  --linearProj 3 --lrExpF 1  --dataset2 cifar100 --task1_nepochs 1 "



def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  
 
