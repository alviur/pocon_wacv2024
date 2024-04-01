from tests import run_main_and_assert


FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 13 --exp_name PFR_epochs --eval_omni_head --head_classifier_lr 10 --classifier_nepochs 500 --lr_factor 3 --lr_patience 20 --sslModel BT --datasets imagenet100 --network resnet18 --num_tasks 4 --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5  --lr_min 5e-7   --kd_method L2  --approach evalOmnihead --projectorArc 2048_2048_2048 --wandblog   --expertArch ResNet18 --pathModelT1 /data/users/agomezvi/gd_pfr/#106/checkpoint/ --loadTask1 --loadExpert --linearProj 3 --lrExpF 1  --dataset2 imagenet100"



#--evalExpert
def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')#112
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/gd_pfr')  # 106
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  # 103
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')  # 112
    # run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/')  # 139

