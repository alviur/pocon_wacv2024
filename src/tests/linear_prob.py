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
pathModelT1: str
    Path to the folder with saved models
    
"""

FAST_LOCAL_TEST_ARGS = "--gpu 0  --port 99 --exp_name eval --eval_omni_head  --head_classifier_lr 5e-2 --classifier_nepochs 200 --num_tasks 10 --loadExpert --lambdaExp 1 --nepochs 0 --ret_nepochs 0 --sslModel BT --datasets cifar100_noTrans --network resnet18  --seed 667 --batch_size 512  --optim_name sgd --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7   --kd_method L2  --approach evalOmnihead  --projectorArc 2048_2048_2048 --dataset2 cifar100  --trans_nepochs 0 --wandblog   --expertArch ResNet18  --task1_nepochs 0  --pathModelT1 /data/users/agomezvi/GD_PFR/cifar100_noTrans_copyOPscratch_POCON_CIFAR100-04012024_165848_870841/ --loadTask1 --linearProj 0 --lrExpF 1"




def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi/GD_PFR')  
 
