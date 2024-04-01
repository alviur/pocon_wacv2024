from tests import run_main_and_assert

# 20 Redo
FAST_LOCAL_TEST_ARGS = "--gpu 0 --port 01 --exp _name PFR_epochs --lambdapRet 1 --lambdaExp 1 --nepochs 500 --ret_nepochs 2000 --updateBackbone 2000 --norm --no_task_boundary_beta 1 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 10 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --eval_omni_head   --kd_method L2  --approach FtOP --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100 --wandblog  --expertArch ResNet9 --mainArch ResNet18 --linearProj 3 --linearProjRet 3 --lrExpF 1 "


def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')#112
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/gd_pfr')  # 106
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  # 103
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')  # 112
    run_main_and_assert(args_line, 0, 0, result_dir='/home/agomezvi/Desktop/tempGD/')  # Local