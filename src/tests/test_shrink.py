from tests import run_main_and_assert

#1000
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --port 18 --exp _name PFR_epochs --lambdapRet 1 --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --trans_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 20 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --eval_fresh_head --kd_method L2  --approach shrink --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100 --wandblog  --expertArch ResNet9 --mainArch ResNet9 --task1_nepochs 1000 --linearProj 3 --linearProjRet 3 --lrExpF 1 "

# 500
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --port 17 --exp _name PFR_epochs --lambdapRet 1 --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --trans_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 50 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --eval_fresh_head --kd_method L2  --approach shrink --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100 --wandblog  --expertArch ResNet18 --mainArch ResNet9 --task1_nepochs 1000 --linearProj 3 --linearProjRet 3 --lrExpF 1 "
#250

FAST_LOCAL_TEST_ARGS = "--gpu 0 --port 16 --exp _name PFR_epochs --lambdapRet 1 --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --trans_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 20 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --eval_fresh_head --kd_method L2  --approach shrink --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100 --wandblog  --expertArch ResNet9 --mainArch ResNet9 --task1_nepochs 250 --linearProj 3 --linearProjRet 3 --lrExpF 1 "

# ResNet34
#250
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --port 19 --exp _name PFR_epochs --lambdapRet 1 --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --trans_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 20 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --eval_fresh_head --kd_method L2  --approach shrink --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100 --wandblog  --expertArch ResNet34 --mainArch ResNet9 --task1_nepochs 500 --linearProj 3 --linearProjRet 3 --lrExpF 1 "


# ================================ D2E  from scratch external dataset ===================================#
##### RESNET18->RESNET18######
# 10 tasks ethereal-tree-567
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --port 10 --exp _name PFR_epochs --lambdapRet 1 --lambdaExp 1 --nepochs 250 --ret_nepochs 4 --trans_nepochs 4  --extImageNet --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 10 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --eval_omni_head   --kd_method L2  --approach d2eOPscratch --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100 --wandblog  --expertArch ResNet18 --mainArch ResNet18 --task1_nepochs 500 --linearProj 3 --linearProjRet 3 --lrExpF 1 "
##### RESNET9->RESNET9######
# 10 tasks cerulean-oath-568
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --port 11 --exp _name PFR_epochs --lambdapRet 1 --lambdaExp 1 --nepochs 250 --ret_nepochs 4 --trans_nepochs 4  --extImageNet --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 10 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --eval_omni_head   --kd_method L2  --approach d2eOPscratch --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100 --wandblog  --expertArch ResNet9 --mainArch ResNet9 --task1_nepochs 500 --linearProj 3 --linearProjRet 3 --lrExpF 1 "


# ================================ Get Taw ===================================#
# --eval_fresh_head
# --eval_omni_head
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 02 --exp_name PFR_epochs --eval_omni_head --lambdapRet 1 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 4 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalCopyOP  --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_d2eOPscratch__name-01132023_164527_914642/ --loadTask1 --linearProj 0 --lrExpF 0.8"
# FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 04 --exp_name PFR_epochs --eval_omni_head  --head_classifier_lr 5e-1  --num_tasks 50 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalOmnihead  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_d2eOPscratch__name-01142023_093533_163246/ --loadTask1 --linearProj 0 --lrExpF 1"

#[56.19, 59.599999999999994, 60.25, 61.06]
# no norm 5e-2 [56.48, 59.77, 60.529999999999994, 62.029999999999994]
# norm 5e-2 [56.51, 59.81, 60.61, 62.050]



#--evalExpert
def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')#112
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/gd_pfr')  # 106
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  # 103
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')  # 112
    #run_main_and_assert(args_line, 0, 0, result_dir='/home/agomezvi/Desktop/tempGD')  # Local