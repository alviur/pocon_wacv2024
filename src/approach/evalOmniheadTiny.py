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
from .sslModels import BarlowTwins
import sys

# appending a path
print(sys.path)
import approach.utilsProj as utilsProj

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.models as models
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
            lambdapRet=1.0,
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
            reInit=False,
            loadExpert=False,
            pathExperts="",
            reInitExpert=False,
            expertArch="ResNet9",
            saveExpert='',
            extImageNet=False,
            expProjSize=512,
            trans_nepochs=0,
            adaptSche=0,
            linearProj=0,
            getResults = False,
            projWarm = 50,
            lrExpF = 1,
            norm = False,
            loadm2 = False,
            sslModel = "BT"

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
        self.lambdapRet = lambdapRet
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
        self.linearProj = linearProj
        self.getResults = getResults
        self.projWarm = projWarm
        self.lrExpF = lrExpF
        self.norm = norm
        self.loadm2 = loadm2
        self.sslModel = sslModel

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
             'lambdapRet': str(self.lambdapRet),
             'lambdaExp': str(self.lambdaExp),
             'classifier_nepochs': str(self.classifier_nepochs),
             'dataset': self.dataset2,
             'projectorArc': self.projectorArc,
             'reInit': self.reInit,
             'reInitExpert': self.reInitExpert
             }
        parameters = pd.DataFrame(data=d, index=[0])




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
                            choices=['ft', 'GD_PFR_PFR', 'GD_PFR', 'GD_PFR_KDalign', 'GD_PFR_contras', 'L2'])
        parser.add_argument("--p2_hid_dim", type=int, default=512)
        parser.add_argument("--task1_nepochs", type=int, default=500)
        parser.add_argument("--pred_like_p2", action="store_true")
        parser.add_argument("--joint", action="store_true")
        parser.add_argument("--diff_lr", action="store_true")
        parser.add_argument("--change_lr_scheduler", action="store_true")
        parser.add_argument("--lambdapRet", default=1.0, type=float)
        parser.add_argument("--lambdaExp", default=1.0, type=float)
        parser.add_argument("--lamb", default=0.01, type=float)
        parser.add_argument("--lambd", default=0.0051, type=float, help='weight on off-diagonal terms')
        parser.add_argument('--projectorArc', default="2048_2048_2048", type=str, help='projector MLP')
        parser.add_argument("--wandblog", action="store_true")
        parser.add_argument("--reInit", action="store_true")
        parser.add_argument("--loadTask1", action="store_true")
        parser.add_argument("--pathModelT1", default="", type=str)
        parser.add_argument("--pathExperts", default="", type=str)
        parser.add_argument("--port", default='11', type=str)
        parser.add_argument("--dataset2", default="cifar100", type=str, choices=['cifar100', 'tiny', 'imagenet100'])
        parser.add_argument("--ret_nepochs", type=int, default=250)
        parser.add_argument("--trans_nepochs", type=int, default=0)
        parser.add_argument("--loadExpert", action="store_true")
        parser.add_argument("--extImageNet", default=False, action="store_true")
        parser.add_argument("--reInitExpert", action="store_true")
        parser.add_argument("--saveExpert", default='', type=str)
        parser.add_argument("--expertArch", default="ResNet9", type=str, choices=['ResNet9', 'ResNet18'])
        parser.add_argument("--sslModel", default="BT", type=str, choices=['BT', 'mocov2'])
        parser.add_argument("--expProjSize", type=int, default=512)
        parser.add_argument('--adaptSche', default=0, type=int, help='projector MLP')
        parser.add_argument("--linearProj", type=int, default=0)
        parser.add_argument("--getResults", action="store_true")
        parser.add_argument("--norm", action="store_true")
        parser.add_argument("--loadm2", action="store_true")
        parser.add_argument("--projWarm", type=int, default=50)
        parser.add_argument("--lrExpF", default=1.0, type=float)


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

        elif self.dataset2 == 'imagenet100':
            input_height = 224
            imagenet_normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
            collate_fn = lightly.data.SimCLRCollateFunction(input_size=input_height, normalize=imagenet_normalize)
            normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Imagenet

        else:
            print("ERROR DATASET")

        _class_lbl = sorted(np.unique(trn_loader.dataset.labels).tolist())
        self._num_classes = len(_class_lbl)

        self.test_transforms = transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalization
        ])

        # self.test_transforms = transforms.Compose([
        #     transforms.Resize(64),  # resize shorter
        #     transforms.CenterCrop(64),  # take center crop
        #     transforms.ToTensor(),
        #     normalization
        # ])

        self.val_transforms = self.test_transforms

        # self.val_transforms = transforms.Compose([
        #     # transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalization
        # ])

        # self.val_transforms = transforms.Compose([
        #     transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.ToTensor(),
        #     normalization
        # ])

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



    def train(self, t, trn_loader, val_loader, output_path):

        print("===================", len(trn_loader.dataset), self.lr, self.expertArch, self.dataset2, 'Adapt sche: ',
              self.adaptSche, "Linear projector:",self.linearProj)

        self._step = 0
        # self._current_task_classes_num = int(self.model.task_cls[t])
        # init at the beginning
        # Create data loaders for data in t
        self.get_data_loaders(trn_loader, val_loader, t)
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:592' + self.port,
                                             world_size=1, rank=0)  # Seed

        # Create model
        if self.sslModel == "BT":
            self.modelFB = BarlowTwins(self.projectorArc, self.batch_size, self.lambd, self.change_lr_scheduler,
                                       self.nepochs, self.diff_lr, self.kd_method, self.lambdapRet, self.lambdaExp,
                                       self.lr, self.expProjSize, self.adaptSche, self.linearProj,
                                       self.linearProj, self.norm, self.expertArch, self.linearProj)

        self.optim_params = self.modelFB.parameters()

        if self.expertArch == 'ResNet18':
            self.modelFB.expertBackbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.modelFB.expertBackbone.fc = nn.Identity()
            self.modelFB.expertBackbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.modelFB.expertBackbone.maxpool = nn.Identity()
        else:
            self.modelFB.expertBackbone = ResNet9()

        self.modelFB.currentExpert = deepcopy(self.modelFB.expertBackbone)

        self.modelFB.projectorExp = utilsProj.prediction_mlp(512, 256, self.linearProj)
        self.modelFB.projectorRet = utilsProj.prediction_mlp(512, 256, self.linearProj)



        path = self.pathModelT1
        print(path)
        modelDict = torch.load(path, map_location='cuda:0')
        self.modelFB = load_backbone_cifar(self.modelFB, modelDict['simsiam'])
        # self.modelFB = load_backbone(self.modelFB, modelDict['state_dict'])
        # self.modelFB.expertBackbone = deepcopy(load_backbone(self.modelFB.expertBackbone, modelDict['simsiam']))

        # import pdb;pdb.set_trace()
        # self.modelFB.expertBackbone = deepcopy(load_backbone(self.modelFB.expertBackbone, modelDict))

        self.modelFB.lamb = self.lamb




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

                # modelT = deepcopy(self.modelFB.backbone).to(self.device)
                # modelT = deepcopy(self.modelFB.expertBackbone).to(self.device)
                modelT = deepcopy(self.modelFB.backbone).to(self.device)
                modelTexp = deepcopy(self.modelFB.backbone).to(self.device)
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

                    if self.norm:
                        r1 = torch.nn.functional.normalize(modelT(img_1.to(self.device)).flatten(start_dim=1))
                    else:
                        r1 = (modelTexp(img_1.to(self.device)).flatten(start_dim=1))
                    # r1 = (modelT(img_1.to(self.device)).flatten(start_dim=1))
                    r1exp = (modelTexp(img_1.to(self.device)).flatten(start_dim=1))
                    z1 = modelTProj(r1)

                    all_z1.append(z1)
                    all_f1.append(r1)
                    if self.modelFB.p2 is not None and self.modelFB.t > 0:
                        all_p1.append(modelTP2(r1))

                    loss = 0.0  # self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2
                    heads = heads_to_evaluate if heads_to_evaluate else self._task_classifiers
                    # headsexp = [self.expertClass]
                    outputs = [h(r1.to(self.device)) for h in heads]
                    # outputsexp = [h(r1exp.to(self.device)) for h in headsexp]
                    # outputsexp = self.expertClass(r1exp.to(self.device))
                    single_task = (heads_to_evaluate is not None) and (len(heads_to_evaluate) == 1)
                    single_taskexp = (heads_to_evaluate is not None) and (len(heads_to_evaluate) == 1)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets, single_task=single_task)
                    # import pdb;
                    # pdb.set_trace()
                    # hits_tawexp, hits_tagexp = self.calculate_metrics(outputsexp, targets, single_task=single_task)
                    hits_tawexp, hits_tagexp = 0,0
                    # Log
                    total_loss += loss * len(targets)  # TODO
                    total_acc_taw += hits_taw.sum().cpu().item()
                    total_acc_tag += hits_tag.sum().cpu().item()
                    # total_acc_tawexp += hits_tawexp.sum().cpu().item()
                    # total_acc_tagexp += hits_tagexp.sum().cpu().item()
                    total_acc_tawexp = 0
                    total_acc_tagexp = 0
                    total_num += len(targets)

            all_z1 = torch.cat(all_z1)
            all_f1 = torch.cat(all_f1)
            if len(all_p1) > 0:
                all_p1 = torch.cat(all_p1)


        self.expertAccu.append(total_acc_tawexp / total_num)
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
        # modelT = deepcopy(self.modelFB.backbone).to(self.device)
        # modelT = deepcopy(self.modelFB.expertBackbone).to(self.device)
        modelT = deepcopy(self.modelFB.backbone).to(self.device)
        modelTexpert = deepcopy(self.modelFB.backbone
                                ).to(self.device)
        for param in modelT.parameters():
            param.requires_grad = False
        for param in modelTexpert.parameters():
            param.requires_grad = False
        modelT.eval();
        modelTexpert.eval()

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
                override_dataset_transform(val_loader.dataset, self.test_transforms) as _ds_val:
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
            # import pdb;
            # pdb.set_trace()

            for img_1, y in _train_loader:

                _xexp = modelTexpert(img_1.to(self.device)).flatten(start_dim=1)
                # _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
                _x = _xexp
                _x = _x.detach();
                _xexp = _xexp.detach()
                y = torch.LongTensor((y - self.model.task_offset[t]).long().cpu()).to(self.device)
                # y = torch.LongTensor((y-25*t).long().cpu()).to(self.device)

                trainX[contBatch:contBatch + trn_loader.batch_size, :] = _x
                trainY[contBatch:contBatch + trn_loader.batch_size] = y
                trainXexp[contBatch:contBatch + trn_loader.batch_size, :] = _xexp
                contBatch += trn_loader.batch_size

            contBatch = 0
            for img_1, y in _val_loader:
                # _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
                _xexp = modelTexpert(img_1.to(self.device)).flatten(start_dim=1)
                _x = _xexp
                _x = _x.detach();
                _xexp = _xexp.detach()
                y = torch.LongTensor((y - self.model.task_offset[t]).long().cpu()).to(self.device)
                # y = torch.LongTensor((y -25*t).long().cpu()).to(self.device)
                valX[contBatch:contBatch + _val_loader.batch_size, :] = _x
                valY[contBatch:contBatch + _val_loader.batch_size] = y
                valXexp[contBatch:contBatch + _val_loader.batch_size, :] = _xexp
                contBatch += _val_loader.batch_size

        if self.norm:

            return torch.nn.functional.normalize(trainX), \
                trainY, \
                torch.nn.functional.normalize(valX), \
                valY, \
                torch.nn.functional.normalize(trainXexp), \
                torch.nn.functional.normalize(valXexp)
        else:
            return trainX, \
                trainY, \
                valX, \
                valY, \
                trainXexp, \
                valXexp

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
        _task_classifier_optimizer = torch.optim.Adam(_task_classifier.parameters(), lr=lr)
        # _task_classifier_optimizerexp = torch.optim.SGD(_task_classifierexp.parameters(), lr=lr, weight_decay=0)

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
                if t>0:
                    #debig python
                    import pdb;
                    pdb.set_trace()
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
                    _x = _x.detach();
                    _xexp = _x.detach()
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

        return _task_classifier

    def _train_classifier_sgd(self, t, trn_loader, val_loader, name='classifier'):
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
        # _task_classifier_optimizer = torch.optim.Adam(_task_classifier.parameters(), lr=lr)
        # _task_classifier_optimizer = torch.optim.SGD(_task_classifier.parameters(), lr=lr)
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
                if t>0:
                    #debig python
                    import pdb;
                    pdb.set_trace()
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
                    _x = _x.detach();
                    _xexp = _x.detach()
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

            if e==60 or e==80 or e==180:
                lr /= self.lr_factor
                print(' lr={:.1e}'.format(lr), end='')
                _task_classifier_optimizer.param_groups[0]['lr'] = lr
                _task_classifier.model.load_state_dict(best_model)




            # Adapt lr
            if val_loss < best_val_loss or best_model is None:
                best_val_loss = val_loss
                best_model = copy.deepcopy(_task_classifier.model.state_dict())
                patience = self.lr_patience
                print('*', end='', flush=True)
            # else:
            #     # print('', end='', flush=True)
            #     patience -= 1
            #     if patience <= 0:
            #         lr /= self.lr_factor
            #         print(' lr={:.1e}'.format(lr), end='')
            #         if lr < self.lr_min:
            #             print(' NO MORE PATIENCE')
            #             break
            #         patience = self.lr_patience
            #         _task_classifier_optimizer.param_groups[0]['lr'] = lr
            #         _task_classifier.model.load_state_dict(best_model)
            # self.logger.log_scalar(task=t, iter=e + 1, name=f"{name}-patience", value=patience, group="train")
            # self.logger.log_scalar(task=t, iter=e + 1, name=f"{name}-lr", value=lr, group="train")
            print()

        time_taken = time.time() - clock0
        _task_classifier.model.load_state_dict(best_model)
        _task_classifier.eval()
        print(f'{name} - Best ACC: {100 * best_val_acc:.1f} time taken: {time_taken:5.1}s')

        return _task_classifier

    def _train_classifierFull(self, t, trn_loader, val_loader, name='classifier'):

        # Extract embeddings

        modelT = deepcopy(self.modelFB.expertBackbone).to(self.device)
        for param in modelT.parameters():
            param.requires_grad = False

        modelT.eval();

        # Send to multiGPU
        modelT = nn.DataParallel(modelT)
        modelT.to(self.device)

        print("Training classifier...")
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
        # _task_classifier_optimizer = torch.optim.Adam(_task_classifier.parameters(), lr=lr)
        _task_classifier_optimizer = torch.optim.SGD(_task_classifier.parameters(), lr=lr, weight_decay=0)
        _task_classifier_optimizerexp = torch.optim.SGD(_task_classifierexp.parameters(), lr=lr)

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

            with override_dataset_transform(trn_loader.dataset, self.val_transforms) as _ds_train, \
                    override_dataset_transform(val_loader.dataset, self.test_transforms) as _ds_val:
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
                    # print("Iterating train loader...")
                    _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
                    _x = _x.detach()
                    # forward pass
                    mlp_preds = _task_classifier(_x.to(self.device))
                    mlp_loss = F.cross_entropy(mlp_preds, y.to(self.device))
                    # update finetune weights
                    mlp_loss.backward()
                    _task_classifier_optimizer.step()
                    _task_classifier_optimizer.zero_grad()
                    train_loss += mlp_loss.item()
                    train_samples += len(y)

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
                    index = 0
                    for img_1, y in _val_loader:
                        # _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
                        _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
                        _x = _x.detach();
                        # forward pass
                        mlp_preds = _task_classifier(_x.to(self.device))
                        mlp_loss = F.cross_entropy(mlp_preds, y.to(self.device))
                        val_loss += mlp_loss.item()
                        n_corr = (mlp_preds.argmax(1) == y.to(self.device)).sum().cpu().item()
                        n_all = y.size()[0]
                        _val_acc = n_corr / n_all
                        # print(f"{self.name} online acc: {train_acc}")
                        acc_correct += n_corr
                        acc_all += n_all

                        val_step += 1
                        index += val_loader.batch_size

                # main validation loss
                val_loss = val_loss / acc_all
                val_acc = acc_correct / acc_all

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                print(
                    f'| Epoch {e} | Train loss: {train_loss:.6f} | Valid loss: {val_loss:.6f} acc: {100 * val_acc:.2f} |',
                    end=''
                )

                if e == 60 or e == 80 or e == 180:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    _task_classifier_optimizer.param_groups[0]['lr'] = lr
                    _task_classifier.model.load_state_dict(best_model)

                # Adapt lr
                if val_loss < best_val_loss or best_model is None:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(_task_classifier.model.state_dict())
                    patience = self.lr_patience
                    print('*', end='', flush=True)


        time_taken = time.time() - clock0
        _task_classifier.model.load_state_dict(best_model)
        _task_classifier.eval()
        print(f'{name} - Best ACC: {100 * best_val_acc:.1f} time taken: {time_taken:5.1}s')

        return _task_classifier

    def train_downstream_classifier(self, t, trn_loader, val_loader, name='downstream-task-classifier'):
        return self._train_classifier(t, trn_loader, val_loader, name)
        # return self._train_classifier_sgd(t, trn_loader, val_loader, name)

        # return self._train_classifierFull(t, trn_loader, val_loader, name)




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
        print("loading", name, name[0:8])
        model.state_dict()[name].copy_(stateDictSaved[name])

    return model

def load_backbone(model, stateDictSaved):
    # own_state = model.state_dict()
    add = 'encoder.'
    # add = 'module.mainBackbone.'
    #add = 'copyExpert.'
    for nameRaw, param in model.state_dict().items():
        name = nameRaw[9:]
        if ((add+name) not in stateDictSaved):
            continue
        print("---loading", add+name)
        # debug python
        # import pdb; pdb.set_trace()

        model.state_dict()[nameRaw[:9]+name].copy_(stateDictSaved[add+name])

    return model

def load_backbone_cifar(model, stateDictSaved):
    # own_state = model.state_dict()

    for name, param in model.state_dict().items():
    # for name, param in stateDictSaved.items():
    #     print(name[:13])
        if (name not in stateDictSaved):
            continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        #     param = param.data

        if (name[0:8]=='backbone') or (name[0:14]=='expertBackbone') or (name[0:13]=='currentExpert'):
        # if (name[0:13] == 'currentExpert'):
            print("---loading", name)
            model.state_dict()[name].copy_(stateDictSaved[name])

    return model


def load_my_state_dictExp(model, stateDictSaved):
    # own_state = model.state_dict()
    for name, param in model.state_dict().items():
        # print(name[15:], name)
        if name not in stateDictSaved:
            continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        #     param = param.data
        print("loading", name, name[15:])
        model.state_dict()[name].copy_(stateDictSaved[name])

    return model


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

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def corrLoss(feat_s, feat_t):
    temperature = 0.5

    batch_size = int(feat_s.size(0) / 4)
    nor_index = (torch.arange(4 * batch_size) % 4 == 0).cuda()
    aug_index = (torch.arange(4 * batch_size) % 4 != 0).cuda()

    f_s = feat_s
    f_s_nor = f_s[nor_index]
    f_s_aug = f_s[aug_index]
    f_s_nor = f_s_nor.unsqueeze(2).expand(-1, -1, 3 * batch_size).transpose(0, 2)
    f_s_aug = f_s_aug.unsqueeze(2).expand(-1, -1, 1 * batch_size)
    s_simi = F.cosine_similarity(f_s_aug, f_s_nor, dim=1)

    f_t_nor = feat_t[nor_index]
    f_t_aug = feat_t[aug_index]
    f_t_nor = f_t_nor.unsqueeze(2).expand(-1, -1, 3 * batch_size).transpose(0, 2)
    f_t_aug = f_t_aug.unsqueeze(2).expand(-1, -1, 1 * batch_size)
    t_simi = F.cosine_similarity(f_t_aug, f_t_nor, dim=1)
    t_simi = t_simi.detach()

    s_simi_log = F.log_softmax(s_simi / temperature, dim=1)
    t_simi_log = F.softmax(t_simi / temperature, dim=1)
    loss_div = F.kl_div(s_simi_log, t_simi_log, reduction='batchmean')
    return loss_div



# https://discuss.pytorch.org/t/should-i-use-model-eval-when-i-freeze-batchnorm-layers-to-finetune/39495/5
def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def set_bn_train(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()