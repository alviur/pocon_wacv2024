from tests import run_main_and_assert

"""
This file contains the tests for the WACV paper experiments.
Parameters
----------
exp_name: str
    Experiment name
eval_omni_head: bool
    Evaluate the omni-head
nepochs: int    
    Number of epochs for adaptation phase
ret_nepochs: int
    Number of epochs for the retrospection phase
task1_Retnepochs: int
    Number of epochs for the reptrospection phase of task 1
head_classifier_lr: float       
    Head classifier learning rate
classifier_nepochs: int 
    Number of epochs for the classifier
lambdapRet: float
    Lambda for the retrospection phase
lr_factor: int  
    Learning rate factor
lr_patience: int    
    Learning rate patience
sslModel: str   
    SSL model, Barlow Twins
datasets: str
    Dataset to use: cifar100_noTrans
network: str
    Network to use: resnet18
num_tasks: int
    Number of tasks
seed: int   
    Seed
batch_size: int 
    Batch size
optim_name: str 
    Optimizer name
lr: float
    Learning rate
momentum: float 
    Momentum
weight_decay: float
    Weight decay
num_workers: int
    Number of workers
hidden_mlp: int
    Hidden MLP
jitter_strength: float
    Jitter strength
lr_min: float
    Minimum learning rate
kd_method: str
    Knowledge distillation method: L2
approach: str
    Approach to use: copyOPscratch, FtOPscratch, D2OPscratch
projectorArc: str
    Projector architecture 
wandblog: bool
    Wandb log
expertArch: str
    Expert architecture: ResNet18, ResNet9
linearProj: int
    Linear projection: select between 1, 2, 3. In the paper, we use 1
lrExpF: int
    Learning rate factor for the expert
dataset2: str
    Dataset 2: cifar100
task1_nepochs: int
    Number of epochs for task 1
    
"""

FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 9 --exp_name POCON_CIFAR100 --eval_omni_head  --nepochs 500 --ret_nepochs 250 --task1_Retnepochs 500 --head_classifier_lr 5e-2 --classifier_nepochs 1 --lambdapRet 25 -lr_factor 3 --lr_patience 20 --sslModel BT --datasets cifar100_noTrans --network resnet18 --num_tasks 10 --seed 667 --batch_size 256  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5  --lr_min 5e-7   --kd_method L2  --approach copyOPscratch --projectorArc 2048_2048_2048 --wandblog   --expertArch ResNet18  --linearProj 3 --lrExpF 1  --dataset2 cifar100 --task1_nepochs 500 "



def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  
 
