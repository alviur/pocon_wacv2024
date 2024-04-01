from tests import run_main_and_assert



#--------Toy example BT loss
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name evalExpert --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 512 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --loadTask1 " \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --expertArch ResNet18 --loadTask1" \
#                        " --approach evalExpert  --port 22 --task1_nepochs 500 --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100" \
#                        " --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_toy_GD_PFR_PFR_epochs-09222022_101749_683472/  --ret_nepochs 500  --wandblog"

FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name PFR_epochs --datasets cifar100_noTrans" \
                       " --network resnet18 --num_tasks 4 --seed 123 --batch_size 512 "\
                       " --nepochs 1 --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 "\
                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
                       " --lr_patience 20 --lr_min 5e-7 --eval_fresh_head   --kd_method GD_PFR --lambdap2 1 --lambdaExp 1" \
                       " --approach evalExpert  --port 03  --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100" \
                       " --ret_nepochs 1 --trans_nepochs 1 --wandblog  --loadExpert --expertArch ResNet18  --task1_nepochs 1"  \
#                       " --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_toy_GD_PFR_PFR_epochs-10122022_100826_806819/ --loadTask1 --adaptSche 1 --extImageNet"

# --eval_fresh_head
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name toy_BarlowLoss_NoShedule_full --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 512 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head   --kd_method GD_PFR_BT_full --lambdap2 1 --lambdaExp 1" \
#                        " --approach toy_GD_PFR  --port 23  --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 --dataset2 cifar100" \
#                        " --ret_nepochs 500  --wandblog --loadExpert" \
#                        " --pathExperts /data3/users/agomezvi/GD_PFR/expertsCifar/"



#@103:--pathExperts /data/users/agomezvi/GD_PFR/resnet9Expert/4tasks_run2/"
#@112:--pathExperts  /data/users/agomezvi/4tasks_run2/"
#@106:--pathExperts   /data/users/agomezvi/gd_pfr/4tasks_run2/"
#@ Local: /media/agomezvi/Alex_hardrive/tmp/4tasks_run2
## 10 tasks experts
#112: /data/users/agomezvi/20tasks
#103: /datatmp/users/agomezvi/10tasks
# train using pretrained Main


# Test
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name eval --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 512 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head   --kd_method L2 --lambdap2 25 --lambdaExp 0" \
#                        " --approach evalBarlowTwins --loadTask1 --port 11 --wandblog --task1_nepochs 500 --head_classifier_lr 5e-2 --projectorArc 2048_2048_2048 --dataset2 cifar100" \
#                        " --evalPath /data/users/agomezvi/cifar100_noTrans_toy_GD_PFR_FT_cos-08232022_143423_465495/"




def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')#112
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/gd_pfr')  # 106
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  # 103
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')  # 112
    #run_main_and_assert(args_line, 0, 0, result_dir='/media/agomezvi/Alex_hardrive/tmp')  # Local
