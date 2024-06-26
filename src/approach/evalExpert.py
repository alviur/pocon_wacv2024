import torch
import time
from torch import nn
from torch.optim import Optimizer
from argparse import ArgumentParser
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from datasets.exemplars_dataset import ExemplarsDataset
from .learning_approach import Learning_Appr
import torch.nn.functional as F
from typing import Optional, Tuple
import torchvision.transforms as transforms
import cv2
from torch.utils.data.dataloader import DataLoader
from datasets.exemplars_selection import override_dataset_transform
from loggers.exp_logger import ExperimentLogger
from networks.loss import cross_entropy
import copy
from copy import deepcopy
# lightly
import pytorch_lightning as pl
import lightly
from lightly.utils import BenchmarkModule
import torchvision
from .joint import JointDataset
import shutil
import itertools
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.models as models
# WandB Import the wandb library
import wandb


class Appr(Learning_Appr):
    """
    Based on the implementation for pytorch-lighting:
    github.com:zlapp/pytorch-lightning-bolts.git
    """

    def __init__(
            self,
            model,
            device,
            nepochs=100,
            lr=0.05,
            lr_min=1e-4,
            lr_factor=3,
            lr_patience=5,
            clipgrad=10000,
            momentum=0,
            wd=1e-6,
            multi_softmax=False,
            wu_nepochs=0,
            wu_lr_factor=1,
            fix_bn=False,
            eval_on_train=False,
            logger=None,
            exemplars_dataset=None,
            # approach params
            warmup_epochs=0,
            lr_warmup_epochs=10,
            hidden_mlp: int = 2048,
            feat_dim: int = 128,
            # maxpool1 = False,
            # first_conv = False,
            # input_height = 32,
            temperature=0.5,
            gaussian_blur=False,
            jitter_strength=0.4,
            optim_name='sgd',
            lars_wrapper=True,
            exclude_bn_bias=False,
            start_lr: float = 0.,
            final_lr: float = 0.,
            classifier_nepochs=20,
            incremental_lr_factor=0.1,
            eval_nepochs=100,
            head_classifier_lr=5e-3,
            head_classifier_min_lr=1e-6,
            head_classifier_lr_patience=3,
            head_classifier_hidden_mlp=2048,
            init_after_each_task=True,
            kd_method='ft',
            p2_hid_dim=512,
            pred_like_p2=False,
            joint=False,
            diff_lr=False,
            change_lr_scheduler=False,
            lambdap2=1.0,
            lambdaExp=1.0,
            task1_nepochs=500,
            wandblog=False,
            loadTask1=False,
            lamb=0.01,
            projectorArc='8192_8192_8192',
            batch_size=512,
            lambd=0.0051,
            dataset2='cifar100',
            pathModelT1='',
            port='11',
            ret_nepochs=500,
            reInit = False,
            loadExpert = False,
            pathExperts = "",
            reInitExpert = False,
            expertArch = "ResNet9",
            saveExpert = '',
            extImageNet = False,
            expProjSize = 512,
            trans_nepochs = 100,
            adaptSche = 2

    ):
        super(Appr, self).__init__(
            model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd, multi_softmax,
            wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset
        )

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.optim_name = optim_name
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.lr_warmup_epochs = lr_warmup_epochs
        self.classifier_nepochs = classifier_nepochs
        self.incremental_lr_factor = incremental_lr_factor
        self.eval_nepochs = eval_nepochs
        self.head_classifier_lr = head_classifier_lr
        self.head_classifier_min_lr = head_classifier_min_lr
        self.head_classifier_lr_patience = head_classifier_lr_patience
        self.head_classifier_hidden_mlp = head_classifier_hidden_mlp
        self.init_after_each_task = init_after_each_task
        self.kd_method = kd_method
        self.p2_hid_dim = p2_hid_dim
        self.pred_like_p2 = pred_like_p2
        self.diff_lr = diff_lr
        self.change_lr_scheduler = change_lr_scheduler
        self.lambdap2 = lambdap2
        self.lambdaExp = lambdaExp
        self.task1_nepochs = task1_nepochs
        self.loadTask1 = loadTask1
        self.projectorArc = projectorArc
        self.batch_size = batch_size
        self.lambd = lambd
        self.dataset2 = dataset2
        self.pathModelT1 = pathModelT1
        self.port = port
        self.ret_nepochs = ret_nepochs
        self.reInit = reInit
        self.loadExpert = loadExpert
        self.pathExperts = pathExperts
        self.reInitExpert = reInitExpert
        self.expertArch = expertArch
        self.saveExpert = saveExpert
        self.extImageNet = extImageNet
        self.expProjSize = expProjSize
        self.trans_nepochs = trans_nepochs
        self.adaptSche = adaptSche

        # Logs
        self.wandblog = wandblog

        # internal vars
        self._step = 0
        self._encoder_emb_dim = 512
        self._task_classifiers = []
        self._task_classifiers_update_step = -1
        self._task_classifiers_update_step = -1
        self._current_task_dataset = None
        self._current_task_classes_num = None
        self._online_train_eval = None
        self._initialized_net = None
        self._tbwriter: SummaryWriter = self.logger.tbwriter

        # Lightly
        self.gpus = [torch.cuda.current_device()]
        self.distributed_backend = 'ddp' if len(self.gpus) > 1 else None

        # LwF lambda
        self.lamb = np.ones((10, 1)) * lamb
        self.expertAccu = []

        # save embeddings
        self.embeddingAvai = np.zeros((10, 1))
        self.trainX = {}
        self.trainXexp = {}
        self.trainY = {}
        self.valX = {}
        self.valXexp = {}
        self.valY = {}

        # Joint
        self.joint = joint
        if self.joint:
            print('Joint training!')
            self.trn_datasets = []
            self.val_datasets = []

        # Wandb for log purposes
        import pandas as pd
        # Load it into a dataframe
        d = {'nepochs': str(nepochs),
             'head_classifier_lr': str(self.head_classifier_lr),
             'task1_nepochs': str(self.task1_nepochs),
             'kd_method': self.kd_method,
             'lambdap2': str(self.lambdap2),
             'lambdaExp': str(self.lambdaExp),
             'classifier_nepochs': str(self.classifier_nepochs),
             'dataset': self.dataset2,
             'projectorArc': self.projectorArc,
             'reInit': self.reInit,
             'reInitExpert': self.reInitExpert
             }
        parameters = pd.DataFrame(data=d, index=[0])

        if self.wandblog:
            wandb.init(project="Cache Networks")
            wandb.log({'Parameters': wandb.Table(dataframe=parameters)})

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=0.4, help="jitter strength")
        # train params
        parser.add_argument("--optim_name", default="adam", type=str, choices=['adam', 'sgd'])
        parser.add_argument("--lars_wrapper", action="store_true", help="apple lars wrapper over optimizer used")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--lr_warmup_epochs", default=10, type=int, help="number of warmup epochs")

        parser.add_argument("--temperature", default=0.5, type=float, help="temperature parameter in training loss")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
        parser.add_argument(
            "--incremental_lr_factor", type=float, default=0.2, help="lr factor for tasks after first one"
        )
        parser.add_argument("--head_classifier_lr", default=1e-4, type=float, help="learning rate for the classifier")
        parser.add_argument(
            "--head_classifier_min_lr", default=1e-6, type=float, help="min learning rate for the classifier"
        )
        parser.add_argument("--head_classifier_lr_patience", default=3, type=int, help="patience for the classifier")
        parser.add_argument(
            "--head_classifier_hidden_mlp", default=2048, type=int, help="number of neurons in hidden classifier layer"
        )
        parser.add_argument("--classifier_nepochs", type=int, default=100, help="Number of epochs for classifier train")
        parser.add_argument("--eval_nepochs", type=int, default=100, help="Evaluate after each N epochs")
        parser.add_argument("--init_after_each_task", action="store_true", help="No FT, init new network each time")

        parser.add_argument("--kd_method", default="ft", type=str,
                            choices=['ft', 'GD_PFR_PFR',  'GD_PFR', 'GD_PFR_BTPFR'])
        parser.add_argument("--p2_hid_dim", type=int, default=512)
        parser.add_argument("--task1_nepochs", type=int, default=500)
        parser.add_argument("--pred_like_p2", action="store_true")
        parser.add_argument("--joint", action="store_true")
        parser.add_argument("--diff_lr", action="store_true")
        parser.add_argument("--change_lr_scheduler", action="store_true")
        parser.add_argument("--lambdap2", default=1.0, type=float)
        parser.add_argument("--lambdaExp", default=1.0, type=float)
        parser.add_argument("--lamb", default=0.01, type=float)
        parser.add_argument("--lambd", default=0.0051, type=float, help='weight on off-diagonal terms')
        parser.add_argument('--projectorArc', default="8192_8192_8192", type=str, help='projector MLP')
        parser.add_argument("--wandblog", action="store_true")
        parser.add_argument("--reInit", action="store_true")
        parser.add_argument("--loadTask1", action="store_true")
        parser.add_argument("--pathModelT1", default="", type=str)
        parser.add_argument("--pathExperts", default="", type=str)
        parser.add_argument("--port", default='11', type=str)
        parser.add_argument("--dataset2", default="cifar100", type=str, choices=['cifar100', 'tiny'])
        parser.add_argument("--ret_nepochs", type=int, default=250)
        parser.add_argument("--trans_nepochs", type=int, default=100)
        parser.add_argument("--loadExpert", action="store_true")
        parser.add_argument("--extImageNet", default=False, action="store_true")
        parser.add_argument("--reInitExpert", action="store_true")
        parser.add_argument("--saveExpert", default='', type=str)
        parser.add_argument("--expertArch", default="ResNet9", type=str, choices=['ResNet9', 'ResNet18'])
        parser.add_argument("--expProjSize", type=int, default=512)
        parser.add_argument('--adaptSche', default=1, type=int, help='projector MLP')



        return parser.parse_known_args(args)

    def get_data_loaders(self, trn_loader, val_loader, t):  # -> Replace _prepare_transformations

        if self.dataset2 == 'tiny':
            input_height = 64
            imagenet_normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            collate_fn = lightly.data.SimCLRCollateFunction(input_size=input_height, normalize=imagenet_normalize)
            normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Imagenet

        elif self.dataset2 == 'cifar100':
            input_height = 32
            cifar_normalize = {'mean': [0.5071, 0.4866, 0.4409], 'std': [0.2009, 0.1984, 0.2023]}
            collate_fn = lightly.data.SimCLRCollateFunction(input_size=input_height, gaussian_blur=0.,
                                                            normalize=cifar_normalize,
                                                            cj_strength=self.jitter_strength)
            normalization = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
        else:
            print("ERROR DATASET")

        _class_lbl = sorted(np.unique(trn_loader.dataset.labels).tolist())
        self._num_classes = len(_class_lbl)

        self.test_transforms = transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalization
        ])
        self.val_transforms = self.test_transforms

        if self.joint:
            # Merge dataset
            self.trn_datasets.append(trn_loader.dataset)
            self.val_datasets.append(val_loader.dataset)
            trn_dset = JointDataset(self.trn_datasets)
            val_dset = JointDataset(self.val_datasets)

            trn_loader = DataLoader(trn_dset,
                                    batch_size=trn_loader.batch_size,
                                    shuffle=True,
                                    num_workers=trn_loader.num_workers,
                                    pin_memory=trn_loader.pin_memory)

            trainJoint = lightly.data.LightlyDataset.from_torch_dataset(trn_loader.dataset)
            self.joint_loader = torch.utils.data.DataLoader(
                trainJoint,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=trn_loader.num_workers
            )

        trainD = lightly.data.LightlyDataset.from_torch_dataset(trn_loader.dataset)

        self.dataloader_train_ssl = torch.utils.data.DataLoader(
            trainD,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=trn_loader.num_workers
        )

        # Imagenet
        if self.extImageNet:

            imagenet_data = torchvision.datasets.ImageFolder('/data/datasets/ILSVRC2012_32/lanczos/')#103
            #imagenet_data = torchvision.datasets.ImageFolder('/data/Datasets/ILSVRC2012_32/lanczos/')
            trainDimagenet = lightly.data.LightlyDataset.from_torch_dataset(imagenet_data)

            self.imagenetData_loader = torch.utils.data.DataLoader(
                trainDimagenet,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True,
                num_workers=trn_loader.num_workers
            )

    def train(self, t, trn_loader, val_loader):

        print("===================", len(trn_loader.dataset),self.lr, self.expertArch,self.dataset2, 'Adapt sche: ',self.adaptSche)


        self._step = 0
        self._current_task_classes_num = int(self.model.task_cls[t])
        if t == 0:
            # init at the beginning
            # Create data loaders for data in t
            self.get_data_loaders(trn_loader, val_loader, t)
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:592' + self.port,
                                                 world_size=1, rank=0)  # Seed

            # Create model
            self.modelFB = BarlowTwins(self.projectorArc, self.batch_size, self.lambd, self.change_lr_scheduler,
                                       self.nepochs, self.diff_lr, self.kd_method, self.lambdap2, self.lambdaExp,self.lr,
                                       self.expProjSize, self.adaptSche)

            self.optim_params = self.modelFB.parameters()

            if self.expertArch == 'ResNet18':
                self.modelFB.expertBackbone = models.resnet18(pretrained=False, zero_init_residual=True)
                self.modelFB.expertBackbone.fc = nn.Identity()
                self.modelFB.expertBackbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
                self.modelFB.expertBackbone.maxpool = nn.Identity()
            else:
                self.modelFB.expertBackbone = ResNet9()

            self.optimizer, self.scheduler = self.modelFB.configure_optimizers()

            # if self.loadTask1:
            #     path = self.pathModelT1 + 'model-task_0.ckpt'
            #     print(path + 'model-task_0.ckpt')
            #     modelDict = torch.load(path, map_location='cuda:0')
            #     # self.modelFB.load_state_dict(modelDict['state_dict'])
            #     self.modelFB = load_my_state_dict(self.modelFB, modelDict['simsiam'])
            #     self.modelFB.backbone = self.modelFB.expertBackbone

            # if self.loadExpert:
            #     path = self.pathExperts
            #     print("Loading Expert: ",path + 'Expert-model-task_'+str(t)+'.ckpt')
            #     modelDict = torch.load(path + 'Expert-model-task_'+str(t)+'.ckpt', map_location='cuda:0')
            #     self.modelFB.expertBackbone = load_my_state_dict(self.modelFB.expertBackbone, modelDict['Expert'])

            self.modelFB.to(self.device)
            self.modelFB.lamb = self.lamb

        else:
            # Create data loaders for data in t
            self.get_data_loaders(trn_loader, val_loader, t)
            self.modelFB.t = t

            # Save old models
            self.modelFB.oldBackbone = deepcopy(self.modelFB.backbone)
            self.modelFB.oldProjector = deepcopy(self.modelFB.projector)

            if self.reInitExpert:

                if self.expertArch == 'ResNet18':
                    #self.modelFB.expertBackbone = deepcopy(self.modelFB.oldBackbone)
                    self.modelFB.expertBackbone = models.resnet18(pretrained=False, zero_init_residual=True)
                    self.modelFB.expertBackbone.fc = nn.Identity()
                    self.modelFB.expertBackbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
                    self.modelFB.expertBackbone.maxpool = nn.Identity()
                else:
                    self.modelFB.expertBackbone = ResNet9()

                # projector
                sizes = [512] + list(map(int, self.projectorArc.split('_')))
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
                self.modelFB.projector = nn.Sequential(*layers)

                # normalization layer for the representations z1 and z2
                self.modelFB.bn = nn.BatchNorm1d(sizes[-1], affine=False)

                if self.modelFB.kd == 'ft':
                    self.modelFB.projectorRet = deepcopy(self.modelFB.projector)
                    self.modelFB.bnExp = nn.BatchNorm1d(sizes[-1], affine=False)
                    self.modelFB.bnRet = nn.BatchNorm1d(sizes[-1], affine=False)
                    self.modelFB.p2 = self.modelFB._prediction_mlp(512, 256)
                    self.modelFB.p2exp = self.modelFB._prediction_mlp(512, 256)
                else:
                    self.modelFB.projectorExp = self.modelFB._prediction_mlp(512, 256)
                    self.modelFB.projectorRet = self.modelFB._prediction_mlp(512, 256)
                    self.modelFB.bnExp = nn.BatchNorm1d(512, affine=False)
                    self.modelFB.bnRet = nn.BatchNorm1d(512, affine=False)
                    self.modelFB.p2 = self.modelFB._prediction_mlp(512, 256)
                    self.modelFB.p2exp = self.modelFB._prediction_mlp(512, 256)

            self.modelFB.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.modelFB.backbone.fc = nn.Identity()
            self.modelFB.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.modelFB.backbone.maxpool = nn.Identity()


            # path = self.pathModelT1 + 'model-task_'+str(t)+'.ckpt'
            # print(path + 'model-task_'+str(t)+'.ckpt')
            # modelDict = torch.load(path, map_location='cuda:0')
            # # self.modelFB.load_state_dict(modelDict['state_dict'])
            # self.modelFB = load_my_state_dict(self.modelFB, modelDict['simsiam'])
            # self.modelFB.backbone = self.modelFB.expertBackbone







            self.optimizer, self.scheduler = self.modelFB.configure_optimizers()

            # empty embeddings
            self.embeddingAvai = np.zeros((10, 1))
            self.trainX = {}
            self.trainY = {}
            self.valX = {}
            self.valY = {}

            self.modelFB.to(self.device)
            self.modelFB.lamb = self.lamb



    def train_loop(self, t, trn_loader, val_loader, epoch):

        if self.extImageNet and self.modelFB.tranfer:
            lenData = len(self.imagenetData_loader)
            data4Train = self.imagenetData_loader

        elif self.extImageNet and self.modelFB.retrospection:
            lenData = len(self.imagenetData_loader)
            data4Train = self.imagenetData_loader

        elif self.joint:
            lenData = len(self.joint_loader)
            data4Train = self.joint_loader
        else:
            lenData = len(self.dataloader_train_ssl)
            data4Train = self.dataloader_train_ssl

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            #len(self.dataloader_train_ssl),
            lenData,
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        self.modelFB.train()

        end = time.time()

        #for i, ((x0, x1), _, _) in enumerate(self.dataloader_train_ssl):
        for i, ((x0, x1), _, _) in enumerate(data4Train):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output and loss
            loss = self.modelFB(x1=x0.to(self.device), x2=x1.to(self.device))

            # wandb.log({"BT loss": loss.item(), "Tasks": 0})

            losses.update(loss.item(), x0.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        if self.scheduler != None:
            self.scheduler.step()
        if self.wandblog:
            lrtrain = self.optimizer.param_groups[0]['lr']
            wandb.log({"lr train": lrtrain})

    # Contains the evaluation code
    def eval(self, t, orig_val_loader, heads_to_evaluate=None):
        with override_dataset_transform(orig_val_loader.dataset, self.test_transforms) as _ds_val:  # no data aug
            val_loader = DataLoader(
                _ds_val,
                batch_size=orig_val_loader.batch_size,
                shuffle=False,
                num_workers=orig_val_loader.num_workers,
                pin_memory=orig_val_loader.pin_memory
            )

            with torch.no_grad():
                total_loss, total_acc_taw, total_acc_tag, total_num, total_acc_tawexp, total_acc_tagexp = 0, 0, 0, 0, 0, 0
                # modelT = deepcopy(self.modelFB.encoder).to(self.device)
                modelT = deepcopy(self.modelFB.backbone).to(self.device)
                modelTexp = deepcopy(self.modelFB.expertBackbone).to(self.device)
                modelTProj = deepcopy(self.modelFB.projector).to(self.device)
                if self.modelFB.p2 is not None and self.modelFB.t > 0:
                    modelTP2 = deepcopy(self.modelFB.p2).to(self.device)
                    modelTP2.eval()
                modelT.eval()
                modelTexp.eval()
                modelTProj.eval()

                # data to save
                all_z1 = []
                all_h1 = []
                all_f1 = []
                all_p1 = []

                for h in self._task_classifiers:
                    h.eval()
                for img_1, targets in val_loader:
                    r1 = modelT(img_1.to(self.device)).flatten(start_dim=1)
                    r1exp = modelTexp(img_1.to(self.device)).flatten(start_dim=1)
                    z1 = modelTProj(r1)

                    all_z1.append(z1)
                    all_f1.append(r1)
                    if self.modelFB.p2 is not None and self.modelFB.t > 0:
                        all_p1.append(modelTP2(r1))

                    loss = 0.0  # self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2
                    heads = heads_to_evaluate if heads_to_evaluate else self._task_classifiers
                    headsexp = [self.expertClass]
                    outputs = [h(r1.to(self.device)) for h in heads]
                    outputsexp = [h(r1exp.to(self.device)) for h in headsexp]
                    #outputsexp = self.expertClass(r1exp.to(self.device))
                    single_task = (heads_to_evaluate is not None) and (len(heads_to_evaluate) == 1)
                    single_taskexp = (heads_to_evaluate is not None) and (len(heads_to_evaluate) == 1)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets, single_task=single_task)
                    hits_tawexp = 0#hits_tawexp, hits_tagexp = self.calculate_metrics(outputsexp, targets, single_task=single_task)

                    # Log
                    total_loss += loss * len(targets)  # TODO
                    total_acc_taw += hits_taw.sum().cpu().item()
                    total_acc_tag += hits_tag.sum().cpu().item()
                    total_acc_tawexp += 0#hits_tawexp.sum().cpu().item()
                    total_acc_tagexp += 0#hits_tagexp.sum().cpu().item()
                    total_num += len(targets)

            all_z1 = torch.cat(all_z1)
            all_f1 = torch.cat(all_f1)
            if len(all_p1) > 0:
                all_p1 = torch.cat(all_p1)

            matrices = (all_z1, all_f1, all_p1)

        # wandb.log({"omni-head acc": 100 * (total_acc_tag / total_num)})


        self.expertAccu.append(total_acc_tawexp / total_num)
        wandb.log({"Omnihead Expert": total_acc_tawexp / total_num, 'Task': t + 1})
        print("---------Expert test accu: ", self.expertAccu)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cosine_similarity(self, a, b):
        b = b.detach()  # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = -1 * (a * b).sum(-1).mean()
        return sim

    # Extract embeddings only once per task
    def get_embeddings(self, t, trn_loader, val_loader):
        # Get backbone
        # modelT = deepcopy(self.modelFB.encoder).cuda()
        modelT = deepcopy(self.modelFB.backbone).to(self.device)
        modelTexpert = deepcopy(self.modelFB.expertBackbone).to(self.device)
        for param in modelT.parameters():
            param.requires_grad = False
        for param in modelTexpert.parameters():
            param.requires_grad = False
        modelT.eval();modelTexpert.eval()

        # Create tensors to store embeddings
        batchFloorT = (len(trn_loader.dataset) // trn_loader.batch_size) * trn_loader.batch_size if \
            (len(trn_loader.dataset) // trn_loader.batch_size) * trn_loader.batch_size != 0 else len(trn_loader.dataset)
        batchFloorV = len(val_loader.dataset)

        trainX = torch.zeros((batchFloorT, self._encoder_emb_dim), dtype=torch.float).to(self.device)
        trainY = torch.zeros(batchFloorT, dtype=torch.long).to(self.device)
        valX = torch.zeros((batchFloorV, self._encoder_emb_dim), dtype=torch.float).to(self.device)
        valY = torch.zeros(batchFloorV, dtype=torch.long).to(self.device)

        trainXexp = torch.zeros((batchFloorT, self._encoder_emb_dim), dtype=torch.float).to(self.device)
        valXexp = torch.zeros((batchFloorV, self._encoder_emb_dim), dtype=torch.float).to(self.device)

        with override_dataset_transform(trn_loader.dataset, self.val_transforms) as _ds_train, \
                override_dataset_transform(val_loader.dataset, self.val_transforms) as _ds_val:
            _train_loader = DataLoader(
                _ds_train,
                batch_size=trn_loader.batch_size,
                shuffle=False,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
                drop_last=True
            )
            _val_loader = DataLoader(
                _ds_val,
                batch_size=val_loader.batch_size,
                shuffle=False,
                num_workers=val_loader.num_workers,
                pin_memory=val_loader.pin_memory
            )

            contBatch = 0
            for img_1, y in _train_loader:
                _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
                _xexp = modelTexpert(img_1.to(self.device)).flatten(start_dim=1)
                _x = _x.detach();_xexp = _xexp.detach()
                y = torch.LongTensor((y - self.model.task_offset[t]).long().cpu()).to(self.device)
                trainX[contBatch:contBatch + trn_loader.batch_size, :] = _x
                trainY[contBatch:contBatch + trn_loader.batch_size] = y
                trainXexp[contBatch:contBatch + trn_loader.batch_size, :] = _xexp
                contBatch += trn_loader.batch_size

            contBatch = 0
            for img_1, y in _val_loader:
                _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
                _xexp = modelTexpert(img_1.to(self.device)).flatten(start_dim=1)
                _x = _x.detach();_xexp = _xexp.detach()
                y = torch.LongTensor((y - self.model.task_offset[t]).long().cpu()).to(self.device)
                valX[contBatch:contBatch + _val_loader.batch_size, :] = _x
                valY[contBatch:contBatch + _val_loader.batch_size] = y
                valXexp[contBatch:contBatch + _val_loader.batch_size, :] = _xexp
                contBatch += _val_loader.batch_size

        return trainX, trainY, valX, valY, trainXexp, valXexp

    def _train_Expert(self, t, trn_loader, val_loader, name='classifier'):

        # prepare classifier
        clock0 = time.time()
        _class_lbl = sorted(np.unique(trn_loader.dataset.labels).tolist())
        _num_classes = len(_class_lbl)
        # MLP
        # _task_classifier = SSLEvaluator(self._encoder_emb_dim, _num_classes, self.hidden_mlp, 0.0)
        # Linear
        _task_classifierexp = SSLEvaluator(self._encoder_emb_dim, _num_classes, 0, 0.0)

        _task_classifierexp.to(self.device)
        lr = self.head_classifier_lr
        _task_classifier_optimizerexp = torch.optim.Adam(_task_classifierexp.parameters(), lr=lr)

        # train on train dataset after learning representation of task
        classifier_train_step = 0
        val_step = 0
        best_val_loss = 1e10
        best_val_acc = 0.0
        patience = self.lr_patience
        _task_classifierexp.train()
        best_model = None

        for e in range(self.classifier_nepochs):

            # train
            train_loss = 0.0
            train_lossexp = 0.0
            train_samples = 0.0
            index = 0

            while index + trn_loader.batch_size <= self.trainX[str(t)].shape[0]:
                _xexp = self.trainXexp[str(t)][index:index + trn_loader.batch_size, :]
                y = self.trainY[str(t)][index:index + trn_loader.batch_size]
                _xexp = _xexp.detach()
                # forward pass
                mlp_predsexp = _task_classifierexp(_xexp.to(self.device))
                mlp_lossexp = F.cross_entropy(mlp_predsexp, y)
                # update finetune weights
                mlp_lossexp.backward()
                _task_classifier_optimizerexp.step()
                _task_classifier_optimizerexp.zero_grad()
                train_lossexp += mlp_lossexp.item()
                train_samples += len(y)

                # val_acc = self._accuracy(mlp_preds, y)
                # pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
                # self.logger.tbwriter.add_scalar(f"t{t}/{name}-loss", mlp_loss, classifier_train_step)
                classifier_train_step += 1
                index += trn_loader.batch_size

            train_lossexp = train_lossexp / train_samples

            # eval on validation
            _task_classifierexp.eval()
            val_lossexp = 0.0
            acc_correctexp = 0
            acc_all = 0
            with torch.no_grad():
                singelite = False if self.valX[str(t)].shape[0] > val_loader.batch_size else True
                index = 0
                while index + val_loader.batch_size < self.valX[str(t)].shape[0] or singelite:
                    _xexp = self.valXexp[str(t)][index:index + val_loader.batch_size, :]
                    y = self.valY[str(t)][index:index + val_loader.batch_size]
                    _xexp = _xexp.detach()
                    # forward pass
                    mlp_predsexp = _task_classifierexp(_xexp.to(self.device))
                    mlp_lossexp = F.cross_entropy(mlp_predsexp, y)
                    val_lossexp += mlp_lossexp.item()
                    n_correxp = (mlp_predsexp.argmax(1) == y).sum().cpu().item()
                    n_all = y.size()[0]
                    _val_accexp = n_correxp / n_all
                    # print(f"{self.name} online acc: {train_acc}")
                    acc_correctexp += n_correxp
                    acc_all += n_all
                    # val_acc = self._accuracy(mlp_preds, y)
                    # pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
                    # self.logger.tbwriter.add_scalar(f"t{t}/{name}-val-loss", mlp_loss, val_step)

                    val_step += 1
                    index += val_loader.batch_size
                    singelite = False

            # main validation loss
            val_lossexp = val_lossexp / acc_all
            val_accexp = acc_correctexp / acc_all
            if val_accexp > best_val_acc:
                best_val_acc = val_accexp

            print(
                f'| Epoch {e} | Train loss: {train_lossexp:.6f} | Valid loss: {val_lossexp:.6f} acc: {100 * val_accexp:.2f} |',
                end=''
            )

            val_loss = val_lossexp

            # Adapt lr
            if val_loss < best_val_loss or best_model is None:
                best_val_loss = val_loss
                best_model = copy.deepcopy(_task_classifierexp.model.state_dict())
                patience = self.lr_patience
                print('*', end='', flush=True)
            else:
                # print('', end='', flush=True)
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print(' NO MORE PATIENCE')
                        break
                    patience = self.lr_patience
                    _task_classifier_optimizerexp.param_groups[0]['lr'] = lr
                    _task_classifierexp.model.load_state_dict(best_model)


        time_taken = time.time() - clock0

        _task_classifierexp.model.load_state_dict(best_model)
        _task_classifierexp.eval()
        print(f'{name} - Best ACC: {100 * best_val_acc:.1f} time taken: {time_taken:5.1}s')
        return _task_classifierexp


    def _train_classifier(self, t, trn_loader, val_loader, name='classifier'):

        # Extract embeddings
        trainX, trainY, valX, valY, trainXexp, valXexp = self.get_embeddings(t, trn_loader, val_loader)
        self.trainX[str(t)] = trainX
        self.trainY[str(t)] = trainY
        self.valX[str(t)] = valX
        self.valY[str(t)] = valY
        self.trainXexp[str(t)] = trainXexp
        self.valXexp[str(t)] = valXexp

        # prepare classifier
        clock0 = time.time()
        _class_lbl = sorted(np.unique(trn_loader.dataset.labels).tolist())
        _num_classes = len(_class_lbl)
        # MLP
        # _task_classifier = SSLEvaluator(self._encoder_emb_dim, _num_classes, self.hidden_mlp, 0.0)
        # Linear
        _task_classifier = SSLEvaluator(self._encoder_emb_dim, _num_classes, 0, 0.0)
        _task_classifierexp = SSLEvaluator(self._encoder_emb_dim, _num_classes, 0, 0.0)

        _task_classifier.to(self.device)
        _task_classifierexp.to(self.device)
        lr = self.head_classifier_lr
        _task_classifier_optimizer = torch.optim.Adam(_task_classifier.parameters(), lr=lr)
        _task_classifier_optimizerexp = torch.optim.Adam(_task_classifierexp.parameters(), lr=lr)

        # train on train dataset after learning representation of task
        classifier_train_step = 0
        val_step = 0
        best_val_loss = 1e10
        best_val_acc = 0.0
        patience = self.lr_patience
        _task_classifier.train()
        _task_classifierexp.train()
        best_model = None

        for e in range(self.classifier_nepochs):

            # train
            train_loss = 0.0
            train_lossexp = 0.0
            train_samples = 0.0
            index = 0

            while index + trn_loader.batch_size <= self.trainX[str(t)].shape[0]:
                _x = self.trainX[str(t)][index:index + trn_loader.batch_size, :]
                y = self.trainY[str(t)][index:index + trn_loader.batch_size]
                _x = _x.detach()
                # forward pass
                mlp_preds = _task_classifier(_x.to(self.device))
                mlp_loss = F.cross_entropy(mlp_preds, y)
                # update finetune weights
                mlp_loss.backward()
                _task_classifier_optimizer.step()
                _task_classifier_optimizer.zero_grad()
                train_loss += mlp_loss.item()
                train_samples += len(y)

                # val_acc = self._accuracy(mlp_preds, y)
                # pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
                # self.logger.tbwriter.add_scalar(f"t{t}/{name}-loss", mlp_loss, classifier_train_step)
                self.logger.log_scalar(
                    task=t, iter=classifier_train_step, name=f"{name}-loss", value=mlp_loss.item(), group="train"
                )
                classifier_train_step += 1
                index += trn_loader.batch_size

            train_loss = train_loss / train_samples

            # eval on validation
            _task_classifier.eval()
            _task_classifierexp.eval()
            val_loss = 0.0
            acc_correct = 0
            acc_all = 0
            with torch.no_grad():
                singelite = False if self.valX[str(t)].shape[0] > val_loader.batch_size else True
                index = 0
                while index + val_loader.batch_size < self.valX[str(t)].shape[0] or singelite:
                    _x = self.valX[str(t)][index:index + val_loader.batch_size, :]
                    _xexp = self.valXexp[str(t)][index:index + val_loader.batch_size, :]
                    y = self.valY[str(t)][index:index + val_loader.batch_size]
                    _x = _x.detach();_xexp = _x.detach()
                    # forward pass
                    mlp_preds = _task_classifier(_x.to(self.device))
                    mlp_loss = F.cross_entropy(mlp_preds, y)
                    val_loss += mlp_loss.item()
                    n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                    n_all = y.size()[0]
                    _val_acc = n_corr / n_all
                    # print(f"{self.name} online acc: {train_acc}")
                    self.logger.log_scalar(task=t, iter=val_step, name=name + '-val-acc', value=_val_acc, group="val")
                    acc_correct += n_corr
                    acc_all += n_all
                    # val_acc = self._accuracy(mlp_preds, y)
                    # pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
                    # self.logger.tbwriter.add_scalar(f"t{t}/{name}-val-loss", mlp_loss, val_step)
                    self.logger.log_scalar(
                        task=t, iter=val_step, name=f"{name}-val-loss", value=mlp_loss.item(), group="val"
                    )
                    val_step += 1
                    index += val_loader.batch_size
                    singelite = False

            # main validation loss
            val_loss = val_loss / acc_all
            val_acc = acc_correct / acc_all

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(
                f'| Epoch {e} | Train loss: {train_loss:.6f} | Valid loss: {val_loss:.6f} acc: {100 * val_acc:.2f} |',
                end=''
            )

            # Adapt lr
            if val_loss < best_val_loss or best_model is None:
                best_val_loss = val_loss
                best_model = copy.deepcopy(_task_classifier.model.state_dict())
                patience = self.lr_patience
                print('*', end='', flush=True)
            else:
                # print('', end='', flush=True)
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print(' NO MORE PATIENCE')
                        break
                    patience = self.lr_patience
                    _task_classifier_optimizer.param_groups[0]['lr'] = lr
                    _task_classifier.model.load_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name=f"{name}-patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name=f"{name}-lr", value=lr, group="train")
            print()

        time_taken = time.time() - clock0
        _task_classifier.model.load_state_dict(best_model)
        _task_classifier.eval()
        print(f'{name} - Best ACC: {100 * best_val_acc:.1f} time taken: {time_taken:5.1}s')

        self.expertClass = self._train_Expert(t, trn_loader, val_loader, name='classifier')

        return _task_classifier

    def train_downstream_classifier(self, t, trn_loader, val_loader, name='downstream-task-classifier'):
        return self._train_classifier(t, trn_loader, val_loader, name)

    def validateKNN(self, train_loader, val_loader, k=200, t=0.1):
        self.model.eval()
        classes = 100
        total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []

        with override_dataset_transform(train_loader.dataset, self.val_transforms) as _ds_train, \
                override_dataset_transform(val_loader.dataset, self.val_transforms) as _ds_val, \
                torch.no_grad():
            batch_size = 512
            trainloader = DataLoader(
                _ds_train,
                batch_size=batch_size,
                shuffle=False,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
                drop_last=True
            )
            testloader = DataLoader(
                _ds_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=val_loader.num_workers,
                pin_memory=val_loader.pin_memory,
                drop_last=True
            )

            trn_batch_size = trainloader.batch_size
            with torch.no_grad():
                # generate feature bank
                for batch_idx, ((inputs, _, _), targets) in enumerate(trainloader):
                    imgs1 = inputs.to(self.device)
                    target = targets.to(self.device)

                    feature = self.encoder(imgs1)
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature.cpu())
                    feature_labels.append(target)

                # [D, N]
                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(self.device)
                # [N]
                # feature_labels = torch.tensor(train_loader.dataset.targets, device=feature_bank.device)
                feature_labels = torch.cat(feature_labels, dim=0)
                # loop test data to predict the label by weighted knn search
                # for batch_idx, data in enumerate(testloader):
                for batch_idx, ((inputs, _, _), targets) in enumerate(testloader):
                    # images, target = data[0].to(self.device), data[1].long().squeeze()
                    images = inputs.to(self.device)
                    target = targets.to(self.device)
                    # images, target = data[0]['data'], data[0]['data_aug'], data[0]['label'].long().squeeze()

                    feature = self.encoder(images)
                    feature = F.normalize(feature, dim=1)  # .cpu()

                    pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes, k, t)

                    total_num += images.size(0)
                    total_top1 += (pred_labels[:, 0] == target).float().sum().item()

            return total_top1 / total_num * 100

    # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
    # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels


class SSLOnlineEvaluator:
    def __init__(self, t, name, encoder, n_input, n_classes, n_hidden, p, device, logger: ExperimentLogger) -> None:
        self.t = t
        self.name = name
        self.logger: ExperimentLogger = logger
        self.encoder = encoder
        self.device = device
        self.model = SSLEvaluator(n_input, n_classes, n_hidden, p)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)

        self._iter = 0
        self._acc_correct = 0
        self._acc_all = 0
        self._eval_iter = 0
        self._acc_eval_correct = 0
        self._acc_eval_all = 0

    def update(self, x, y):
        y = torch.LongTensor(y).to(self.device)
        with torch.no_grad():
            representations = self.encoder(x.to(self.device))
        representations = representations.detach()  # don't backprop through encoder
        # forward pass
        self.model.train()
        mlp_preds = self.model(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
        n_all = y.size()[0]
        train_acc = n_corr / n_all
        # print(f"{self.name} online acc: {train_acc}")
        self.logger.log_scalar(task=self.t, iter=self._iter, name=self.name + '-train', value=train_acc, group="train")
        self._acc_correct += n_corr
        self._acc_all += n_all
        self._iter += 1

    def eval(self, x, y):
        y = torch.LongTensor(y).to(self.device)
        self.model.eval()
        with torch.no_grad():
            representations = self.encoder(x.to(self.device))
            representations = representations.detach()  # don't backprop through encoder
            # forward pass  d
            mlp_preds = self.model(representations)

            # log metrics
            n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
            n_all = y.size()[0]
            _acc = n_corr / n_all
            self.logger.log_scalar(
                task=self.t, iter=self._iter, name=self.name + '-validation', value=_acc, group="valid"
            )
        self._acc_eval_correct += n_corr
        self._acc_eval_all += n_all
        self._eval_iter += 1

    def acc(self):
        if self._acc_all == 0:
            return 0.0
        return self._acc_correct / self._acc_all

    def acc_eval(self):
        if self._acc_eval_all == 0:
            return 0.0
        return self._acc_eval_correct / self._acc_eval_all

    def acc_reset(self):
        self._acc_correct = 0
        self._acc_all = 0
        self._eval_iter = 0
        self._acc_eval_correct = 0
        self._acc_eval_all = 0


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.out_features = n_classes  # for *head* compability
        if n_hidden is None or n_hidden == 0:
            # use linear classifier
            self.model = nn.Sequential(nn.Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.model(x)
        return logits


class MLP(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_size: int = 4096, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(
            self,
            encoder: Optional[nn.Module] = None,
            input_dim: int = 2048,
            hidden_size: int = 4096,
            output_dim: int = 256,
    ) -> None:
        super().__init__()

        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(input_dim, hidden_size, output_dim)
        # Predictor
        self.predictor = MLP(output_dim, hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


class SimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR
    Transform::
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform
        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
            self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength, 0.8 * self.jitter_strength, 0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose(
            [transforms.RandomResizedCrop(self.input_height),
             transforms.RandomHorizontalFlip(), self.final_transform]
        )

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)


class SimCLREvalDataTransform(SimCLRTrainDataTransform):
    """
    Transforms for SimCLR
    Transform::
        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform
        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
            self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ):
        super().__init__(
            normalize=normalize,
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose(
            [
                # transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                # transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )


class SimCLRFinetuneTransform(object):
    def __init__(
            self,
            input_height: int = 224,
            jitter_strength: float = 1.,
            normalize=None,
            eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        if not eval_transform:
            data_transforms = [
                transforms.RandomResizedCrop(size=self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]
        else:
            data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height)
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class LARSWrapper(object):
    """
    Wrapper that adds LARS scheduling to any optimizer. This helps stability with huge batch sizes.
    References:
    - https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
    """

    def __init__(self, optimizer, eta=0.02, clip=True, eps=1e-8):
        """
        Args:
            optimizer: torch optimizer
            eta: LARS coefficient (trust)
            clip: True to clip LR
            eps: adaptive_lr stability coefficient
        """
        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group
        self.__setstate__ = self.optim.__setstate__
        self.__getstate__ = self.optim.__getstate__
        self.__repr__ = self.optim.__repr__

    @property
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

    @property
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get('weight_decay', 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group['weight_decay'] = 0

            # update the parameters
            [self.update_p(p, group, weight_decay) for p in group['params'] if p.grad is not None]

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group['lr'], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr


class BarlowTwins(nn.Module):
    def __init__(self, projectorArc, batch_size, lambd, change_lr_scheduler, maxEpochs, diff_lr, kd_method, lambdap2,
                 lambdaExp, lr, expProjSize, adaptSche):
        super().__init__()
        ### Barlow Twins params ###
        self.projectorArc = projectorArc
        print("-----------------------------", self.projectorArc)
        self.batch_size = 512
        self.lambd = lambd
        self.scale_loss = 0.025

        ### Continual learning parameters ###
        self.kd = kd_method
        self.t = 0
        self.oldModel = None
        self.oldModelFull = None
        self._task_classifiers = None
        self.lamb = None
        self.lambdap2 = lambdap2
        self.lambdaExp = lambdaExp
        self.criterion = nn.CosineSimilarity(dim=1)
        self.retrospection = False
        self.transfer = False

        self.p2 = self._prediction_mlp(512, 256)

        ### Training params
        self.change_lr_scheduler = change_lr_scheduler
        self.maxEpochs = maxEpochs
        self.diff_lr = diff_lr
        self.base_lr = lr  # 0.01
        self.lars_wrapper = False

        # log
        self.wandblog = False

        # Architecture
        self.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()


        # Distillation projectors
        self.expProjSize = expProjSize
        self.adaptSche = adaptSche
        self.adaptW = 1

        # projector
        sizes = [512] + list(map(int, self.projectorArc.split('_')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # Projectors
        if self.kd == 'ft':
            self.p2 = self._prediction_mlp(512, 256)
            self.p2exp = self._prediction_mlp(512, 256)
            self.bnExp = nn.BatchNorm1d(sizes[-1], affine=False)
            self.bnRet = nn.BatchNorm1d(512, affine=False)
            self.projectorExp = deepcopy(self.projector)
            self.projectorExp = deepcopy(self.projector)
            self.projectorRet = deepcopy(self.projector)
        else:
            self.p2 = self._prediction_mlp(512, 256)
            self.p2exp = self._prediction_mlp(512, 256)
            self.bnExp = nn.BatchNorm1d(self.expProjSize, affine=False)
            self.bnRet = nn.BatchNorm1d(self.expProjSize, affine=False)
            self.projectorExp = self._prediction_mlp(self.expProjSize, 256)
            self.projectorRet = self._prediction_mlp(self.expProjSize, int(self.expProjSize/2))




        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


        ## EWC ##
        if self.kd == 'EWC' or self.kd == 'EWC_p2':
            self.sampling_type = 'contrastive'  # ce, contrastive,contrastive_lwf
            feat_ext = self.backbone
            # Store current parameters as the initial parameters before first task starts
            self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
            # Store fisher information weight importance
            self.fisher = {n: torch.zeros(p.shape).cuda() for n, p in feat_ext.named_parameters()
                           if p.requires_grad}
            self.num_samples = -1
            self.alpha = 0.5
            self.lossP2 = 0

    def forward(self, x1, x2):

        if(not(self.retrospection) and not(self.tranfer)):
            if(self.kd == 'ft'):
                f1 = torch.squeeze(self.backbone(x1))
                f2 = torch.squeeze(self.backbone(x2))
            else:
                f1 = torch.squeeze(self.expertBackbone(x1))
                f2 = torch.squeeze(self.expertBackbone(x2))


            z1 = self.projector(f1)
            z2 = self.projector(f2)
            # z1 = (f1)
            # z2 = (f2)

            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(x1.shape[0])
            torch.distributed.all_reduce(c)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            # loss = self.scale_loss*(on_diag + self.lambd * off_diag)
            loss = (0.025 * (on_diag + 0.0051 * off_diag))

            wandb.log({"loss expert": loss.item()})

        elif self.tranfer and not(self.retrospection):
            # print("I am doing transfer")

            f1 = torch.squeeze(self.backbone(x1)).detach()
            f2 = torch.squeeze(self.backbone(x2)).detach()

            f1Expert = torch.squeeze(self.expertBackbone(x1))
            f2Expert = torch.squeeze(self.expertBackbone(x2))

            # Map to old task
            z1 = (f1)
            z2 = self.projectorExp(f1Expert)

            # empirical cross-correlation matrix
            c = self.bnExp(z1).T @ self.bnExp(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(x1.shape[0])
            torch.distributed.all_reduce(c)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()

            lossKD1 = (0.025 * (on_diag + 0.0051 * off_diag))

            # Map to new task
            z1_2 = (f2)
            z2_3 = self.projectorExp(f2Expert)

            # empirical cross-correlation matrix
            c = self.bnExp(z1_2).T @ self.bnExp(z2_3)

            # sum the cross-correlation matrix between all gpus
            c.div_(x1.shape[0])
            torch.distributed.all_reduce(c)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            # loss = self.scale_loss*(on_diag + self.lambd * off_diag)
            lossKD2 = (0.025 * (on_diag + 0.0051 * off_diag))

            loss = lossKD1 + lossKD2
            wandb.log({"loss transfer": loss.item()})
            lossRetro = 0



        elif self.kd == 'GD_PFR' and self.retrospection:

            f1 = torch.squeeze(self.backbone(x1))
            f2 = torch.squeeze(self.backbone(x2))

            f1Expert = torch.squeeze(self.expertBackbone(x1)).detach()
            f2Expert = torch.squeeze(self.expertBackbone(x2)).detach()

            # Map to old task
            z1 = self.projectorExp(f1)
            #z1 = (f1)
            z2 = (f1Expert)

            # empirical cross-correlation matrix
            c = self.bnExp(z1).T @ self.bnExp(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(x1.shape[0])
            torch.distributed.all_reduce(c)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()

            lossKD1 = (0.025 * (on_diag + 0.0051 * off_diag))

            # Map to new task
            z1_2 = self.projectorExp(f2)
            #z1_2 = (f2)
            z2_3 = (f2Expert)

            # empirical cross-correlation matrix
            c = self.bnExp(z1_2).T @ self.bnExp(z2_3)

            # sum the cross-correlation matrix between all gpus
            c.div_(x1.shape[0])
            torch.distributed.all_reduce(c)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            # loss = self.scale_loss*(on_diag + self.lambd * off_diag)
            lossKD2 = (0.025 * (on_diag + 0.0051 * off_diag))

            lossAdapt = (self.adaptSche/self.adaptW)*(lossKD1 + lossKD2)

            if self.t > 0:
                f1old = torch.squeeze(self.oldBackbone(x1)).detach()
                f2old = torch.squeeze(self.oldBackbone(x2)).detach()

                # Map to old task

                z1 = self.projectorRet(f1)
                #z1 = (f1)
                z2 = (f1old)

                # empirical cross-correlation matrix
                c = self.bnRet(z1).T @ self.bnRet(z2)

                # sum the cross-correlation matrix between all gpus
                c.div_(x1.shape[0])
                torch.distributed.all_reduce(c)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                lossold1 = (0.025 * (on_diag + 0.0051 * off_diag))

                # Map to old task
                z1_2 = self.projectorRet(f2)
                #z1_2 = (f2)
                z2_3 = (f2old)

                # empirical cross-correlation matrix
                c = self.bnRet(z1_2).T @ self.bnRet(z2_3)

                # sum the cross-correlation matrix between all gpus
                c.div_(x1.shape[0])
                torch.distributed.all_reduce(c)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                lossold2 = (0.025 * (on_diag + 0.0051 * off_diag))

                lossRetro = self.lambdaExp * (lossold1 + lossold2)

                wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": lossRetro.item(), "LambdaExp": self.lambdaExp})

            else:
                lossRetro = 0
                wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": 0})

            loss = lossAdapt + lossRetro

        elif self.kd == 'GD_PFR_PFR' and self.retrospection:
            f1 = torch.squeeze(self.backbone(x1))
            f2 = torch.squeeze(self.backbone(x2))

            f1Expert = torch.squeeze(self.expertBackbone(x1)).detach()
            f2Expert = torch.squeeze(self.expertBackbone(x2)).detach()

            p = self.projectorExp(f1)
            p_2 = self.projectorExp(f2)


            lossAdapt = (-(self.criterion(p, f1Expert.detach()).mean()
                                            + self.criterion(p_2, f2Expert.detach()).mean()) * 0.5)


            if self.t > 0:
                f1old = torch.squeeze(self.oldBackbone(x1)).detach()
                f2old = torch.squeeze(self.oldBackbone(x2)).detach()

                p2_1 = self.projectorRet(f1)
                p2_2 = self.projectorRet(f2)

                lossRetro = self.lambdaExp * (-(self.criterion(p2_1, f1old.detach()).mean()
                                                + self.criterion(p2_2, f2old.detach()).mean()) * 0.5)

                wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": lossRetro.item(), "LambdaExp": self.lambdaExp})

            else:
                lossRetro = 0
                wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": 0})

            loss = lossAdapt + lossRetro


        elif self.kd == 'GD_PFR_BTPFR' and self.retrospection:

            if self.t > 0:

                f1 = torch.squeeze(self.backbone(x1))
                f2 = torch.squeeze(self.backbone(x2))

                f1Expert = torch.squeeze(self.expertBackbone(x1)).detach()
                f2Expert = torch.squeeze(self.expertBackbone(x2)).detach()

                p = self.projectorExp(f1)
                p_2 = self.projectorExp(f2)


                lossAdapt = (-(self.criterion(p, f1Expert.detach()).mean()
                                                + self.criterion(p_2, f2Expert.detach()).mean()) * 0.5)
            else:
                f1 = torch.squeeze(self.backbone(x1))
                f2 = torch.squeeze(self.backbone(x2))

                f1Expert = torch.squeeze(self.expertBackbone(x1)).detach()
                f2Expert = torch.squeeze(self.expertBackbone(x2)).detach()

                # Map to old task
                z1 = self.projectorExp(f1)
                z2 = (f1Expert)

                # empirical cross-correlation matrix
                c = self.bnExp(z1).T @ self.bnExp(z2)

                # sum the cross-correlation matrix between all gpus
                c.div_(x1.shape[0])
                torch.distributed.all_reduce(c)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()

                lossKD1 = (0.025 * (on_diag + 0.0051 * off_diag))

                # Map to new task
                z1_2 = self.projectorExp(f2)
                z2_3 = (f2Expert)

                # empirical cross-correlation matrix
                c = self.bnExp(z1_2).T @ self.bnExp(z2_3)

                # sum the cross-correlation matrix between all gpus
                c.div_(x1.shape[0])
                torch.distributed.all_reduce(c)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                # loss = self.scale_loss*(on_diag + self.lambd * off_diag)
                lossKD2 = (0.025 * (on_diag + 0.0051 * off_diag))

                lossAdapt = lossKD1 + lossKD2


            if self.t > 0:
                f1old = torch.squeeze(self.oldBackbone(x1)).detach()
                f2old = torch.squeeze(self.oldBackbone(x2)).detach()

                p2_1 = self.projectorRet(f1)
                p2_2 = self.projectorRet(f2)

                lossRetro = self.lambdaExp * (-(self.criterion(p2_1, f1old.detach()).mean()
                                                + self.criterion(p2_2, f2old.detach()).mean()) * 0.5)

                wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": lossRetro.item(), "LambdaExp": self.lambdaExp})

            else:
                lossRetro = 0
                wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": 0})

            loss = lossAdapt + lossRetro

        else:

            loss = 0




        return loss

    def configure_optimizers(self):

        if self.t < 10:

            params = list(self.backbone.parameters())
            params += list(self.projector.parameters())
            params += list(self.expertBackbone.parameters())
            params += list(self.projectorExp.parameters())
            params += list(self.projectorRet.parameters())


            optim = torch.optim.SGD(params, lr=self.base_lr, momentum=0.9, weight_decay=5e-4)
            max_steps = (int(2.3 * self.maxEpochs)) if self.change_lr_scheduler else self.maxEpochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)
            self.scheduler = scheduler

            # if self.lars_wrapper:
            #     optim = LARSWrapper(
            #         optim,
            #         eta=0.02,  # trust coefficient
            #         clip=True
            #     )

            #return optim, scheduler
            return optim, None

        else:
            self.base_lr = self.base_lr*0.8

            params = list(self.backbone.parameters())
            params += list(self.projector.parameters())
            params += list(self.expertBackbone.parameters())
            params += list(self.projectorExp.parameters())
            params += list(self.projectorRet.parameters())

            optim = torch.optim.SGD(params, lr=self.base_lr, momentum=0.9, weight_decay=5e-4)
            max_steps = (int(2.3 * self.maxEpochs)) if self.change_lr_scheduler else self.maxEpochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)
            self.scheduler = scheduler

            # if self.lars_wrapper:
            #     optim = LARSWrapper(
            #         optim,
            #         eta=0.02,  # trust coefficient
            #         clip=True
            #     )

            # return optim, scheduler
            return optim, None


    # def configure_optimizers(self):
    #
    #     if self.t < -1:
    #         params = list(self.backbone.parameters())
    #         params += list(self.projector.parameters())
    #         params += list(self.projectorExp.parameters())
    #         params += list(self.projectorRet.parameters())
    #         params += list(self.p2.parameters())
    #
    #         optim = torch.optim.SGD(params, lr=self.base_lr, momentum=0.9, weight_decay=5e-4)
    #         max_steps = (int(2.3 * self.maxEpochs)) if self.change_lr_scheduler else self.maxEpochs
    #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)
    #         self.scheduler = scheduler
    #
    #         # if self.lars_wrapper:
    #         #     optim = LARSWrapper(
    #         #         optim,
    #         #         eta=0.02,  # trust coefficient
    #         #         clip=True
    #         #     )
    #
    #
    #         return optim, scheduler
    #
    #
    #     else:
    #         lr = self.base_lr
    #         params = [
    #             {'params': self.backbone.parameters(), 'lr': lr*0.01 if self.diff_lr else 0.5*0.8*lr},
    #             #{'params': self.backbone.parameters(), 'lr': lr * 0.01 if self.diff_lr else 0.8 * lr},
    #             {'params': self.projector.parameters(), 'lr': lr*0.3 if self.diff_lr else 0.8*lr},
    #             {'params': self.projectorExp.parameters(), 'lr': lr * 0.3 if self.diff_lr else 0.8 * lr},
    #             {'params': self.projectorRet.parameters(), 'lr': lr * 0.3 if self.diff_lr else 0.8 * lr},
    #         ]
    #
    #         print('Optimizer lr: ', [d['lr'] for d in params])
    #         optim = torch.optim.SGD(params, lr=lr * 0.8, momentum=0.9, weight_decay=5e-4)
    #         return optim, None



    def _prediction_mlp(self, in_dims: int, h_dims: int):

        prediction = nn.Sequential(nn.Linear(in_dims, h_dims, bias=False),
                                   nn.BatchNorm1d(h_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(h_dims, in_dims))  # output layer

        return prediction

    def criterionEWC(self, t):
        loss = 0
        if t > 0:
            loss_reg = 0
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.backbone.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(
                        self.fisher[n].cuda() * (p - self.older_params[n].cuda()).pow(2)) / 2
            loss += self.lambdap2 * loss_reg

        return loss

    def post_train_process(self, t, trn_loader):
        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.backbone.named_parameters() if
                             p.requires_grad}

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader, t)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).cuda()
            else:
                self.fisher[n] = (self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n])

    def compute_fisher_matrix_diag(self, trn_loader, t):

        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).cuda() for n, p in
                  self.backbone.named_parameters()
                  if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to compute the fisher information
        modelT = deepcopy(self.backbone).cuda()
        for param in modelT.parameters():
            param.requires_grad = True
        modelT.train()
        # for images, targets in itertools.islice(trn_loader, n_samples_batches):

        for ((inputs, inputsAug, _), targets) in itertools.islice(trn_loader, n_samples_batches):

            if self.sampling_type == 'ce':

                _x1 = torch.squeeze(self.resnet_simsiam.backbone(inputs.cuda()))
                # Forward old model
                classHead = deepcopy(self._task_classifiers[self.t]).cuda()
                for param in classHead.parameters():
                    param.requires_grad = True
                outputs = classHead(_x1)

                # Use the labels to compute the gradients based on the CE-loss with the ground truth
                preds = (targets - self.t * 10).cuda()
                filtered_tensor = outputs[~torch.any(outputs.isnan(), dim=1)]
                filtered_labels = preds[~torch.any(outputs.isnan(), dim=1)]
                loss = torch.nn.functional.cross_entropy((filtered_tensor), filtered_labels)

            elif self.sampling_type == 'contrastive':

                f1 = torch.squeeze(self.backbone(inputs.cuda()))
                f2 = torch.squeeze(self.backbone(inputsAug.cuda()))
                z1 = self.projector(f1)
                z2 = self.projector(f2)

                # empirical cross-correlation matrix
                c = self.bn(z1).T @ self.bn(z2)

                # sum the cross-correlation matrix between all gpus
                c.div_(self.batch_size)
                torch.distributed.all_reduce(c)

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                loss = self.scale_loss * (on_diag + self.lambd * off_diag)


            elif self.sampling_type == 'contrastive_lwf':

                f1 = torch.squeeze(self.backbone(inputs.cuda()))
                f2 = torch.squeeze(self.backbone(inputsAug.cuda()))
                z1 = self.projector(f1)
                z2 = self.projector(f2)

                p1 = self.predictor(
                    z1)  # NxC
                p2 = self.predictor(
                    z2)  # NxC

                loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5

                if t > 0:
                    f1Old = torch.squeeze(self.oldBackbone(inputs.cuda()))
                    f2Old = torch.squeeze(self.oldBackbone(inputsAug.cuda()))
                    z1Old = self.oldProjector(f1Old)
                    z2Old = self.oldProjector(f2Old)

                    p2_1 = self.p2(z1)
                    p2_2 = self.p2(z2)

                    lossKD = 1 * (-(self.criterion(p2_1, z1Old.detach()).mean()
                                    + self.criterion(p2_2, z2Old.detach()).mean()) * 0.5)

                    loss += lossKD

            self.backbone.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self.backbone.named_parameters():
                if p.grad is not None and torch.isnan(p).any().cpu().numpy().item() is not True:
                    # fisher[n] += p.grad.pow(2) * len(targets)
                    fisher[n] += (p.grad * pow(len(targets), 0.5)).pow(2)
                    # if torch.isnan(p).any(): print("NAN!!!!!!");

        # Apply mean across all samples
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_my_state_dict(model, stateDictSaved):
    # own_state = model.state_dict()
    for name, param in model.state_dict().items():
        if name not in stateDictSaved:
            continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        #     param = param.data
        print("loading", name, name[15:])
        model.state_dict()[name].copy_(stateDictSaved[name])

    return model

def load_my_state_dictExp(model, stateDictSaved):
    # own_state = model.state_dict()
    for name, param in model.state_dict().items():
        print(name[15:],name)
        if name not in stateDictSaved:
            continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        #     param = param.data
        print("loading", name, name[15:])
        model.state_dict()[name].copy_(stateDictSaved[name])

    return model


class expertNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 512)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out