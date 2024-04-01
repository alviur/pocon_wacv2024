from tests import run_main_and_assert

# ================================ Eval ===================================#

# Eval R18-R18 --getExpert --norm
#Expert
FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 91 --getExpert --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 50 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalCurrentTask  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/pocon/models/cifar100/cassle/r18/barlow-cifar100-cassle-split:50-2023-04-02_18:57:49/ --loadTask1 --linearProj 0 --lrExpF 1"
# main
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 93  --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 4 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalCurrentTask  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/cifar100_noTrans_pfr_PFR_10tasks_trajectories-05132023_194618_156529/ --loadTask1 --linearProj 0 --lrExpF 1"

# 10 tasks
#Expert
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 90 --getExpert --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 10 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalCurrentTask  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_FtOP__name-05132023_194010_559783/ --loadTask1 --linearProj 0 --lrExpF 1"
# main
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 90  --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 10 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalCurrentTask  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_FtOP__name-05132023_194010_559783/ --loadTask1 --linearProj 0 --lrExpF 1"


# 50 tasks
#Expert
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 90 --getExpert --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 50 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalCurrentTask  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/pocon/analysis/50_tasks/pocon/ --loadTask1 --linearProj 0 --lrExpF 1"
# main
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 90  --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 50 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalCurrentTask  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/pocon/analysis/50_tasks/pfr/ --loadTask1 --linearProj 0 --lrExpF 1"















# Eval R9-R9
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 11 --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 50 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalOmnihead  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet9  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_copyOP__name-04242023_103642_815912/ --loadTask1 --linearProj 0 --lrExpF 1"

# Eval R18-R9
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 11 --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 50 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalOmnihead  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_FtOP__name-04212023_174618_549730/ --loadTask1 --linearProj 0 --lrExpF 1"


# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 12 --exp_name PFR_epochs --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 10 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalOmnihead --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/CIFAR100/d2e/10tasks/r18_r18_500ret/ --loadTask1 --linearProj 0 --lrExpF 1"






#--evalExpert
def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')#112
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/gd_pfr')  # 106
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  # 103
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  # 139
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/')  # 112
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/miniconda3/')  # 138
    # run_main_and_assert(args_line, 0, 0, result_dir='/home/agomezvi/D
    # esktop/tempGD')  # Local