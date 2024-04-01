import torch
import torch.nn as nn
import approach.utilsProj as utilsProj
import torchvision.models as models
import torch.nn.functional as F
import torch.distributed as dist
# WandB Import the wandb library
import wandb

# Barlow twins
class BarlowTwins(nn.Module):
    def __init__(self, projectorArc, batch_size, lambd, change_lr_scheduler, maxEpochs, diff_lr, kd_method, lambdapRet,
                 lambdaExp, lr, expProjSize, adaptSche, linearProj, linearProjRet, norm, mainArch,lambVect):
        super().__init__()
        ### Barlow Twins params ###
        self.projectorArc = projectorArc
        print("-----------------------------", self.projectorArc)
        self.batch_size = batch_size
        self.lambd = lambd
        self.scale_loss = 0.025
        self.linearProj = linearProj
        self.linearProjRet = linearProjRet
        self.norm = norm
        self.mainArch = mainArch
        self.lamb = lambVect

        ### Continual learning parameters ###
        self.kd = kd_method
        self.t = 0
        self.oldModel = None
        self.oldModelFull = None
        self._task_classifiers = None

        self.lambdapRet = lambdapRet
        self.lambdaExp = lambdaExp
        self.criterion = nn.CosineSimilarity(dim=1)
        self.retrospection = False
        self.transfer = False

        self.num_features = 512
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

        self.p2 = utilsProj.prediction_mlp(512, 256, self.linearProj)

        ### Training params
        self.change_lr_scheduler = change_lr_scheduler
        self.maxEpochs = maxEpochs
        self.diff_lr = diff_lr
        self.base_lr = lr  # 0.01
        self.lars_wrapper = False
        self.switchHead = True

        # log
        self.wandblog = False

        # Architecture
        if self.mainArch == "ResNet18":
            self.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.backbone.fc = nn.Identity()
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.backbone.maxpool = nn.Identity()
        elif self.mainArch == "ResNet18Com":
            self.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        elif self.mainArch == "ResNet9":
            self.backbone = ResNet9()

        elif self.mainArch == "mobilenet":
            from torchvision.models import mobilenet_v2
            self.backbone = mobilenet_v2(pretrained=False)
            self.backbone.classifier = nn.Linear(1280, 512)

        elif self.mainArch == "shufflenet":
            from torchvision.models import shufflenet_v2_x0_5
            self.backbone = shufflenet_v2_x0_5(pretrained=False)
            self.backbone.fc = nn.Identity()#nn.Linear(1024, 512)
            # self.backbone.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
            # self.backbone.maxpool = nn.Identity()

        # Distillation projectors
        self.expProjSize = expProjSize
        self.adaptSche = adaptSche
        self.adaptW = 1

        self.K = 65536
        self.register_buffer("queue", torch.randn(512, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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
        self.p2 = utilsProj.prediction_mlp(512, 256, self.linearProj)
        self.p2exp = utilsProj.prediction_mlp(512, 256, self.linearProj)
        self.bnExp = nn.BatchNorm1d(self.expProjSize, affine=False)
        self.bnRet = nn.BatchNorm1d(self.expProjSize, affine=False)
        self.projectorExp = utilsProj.prediction_mlp(self.expProjSize, 256, self.linearProj)
        self.projectorRet = utilsProj.prediction_mlp(self.expProjSize, int(self.expProjSize / 2), self.linearProjRet)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


    def forward(self, x1, x2):

        if (not (self.retrospection) and not (self.tranfer)):

            if self.norm and self.t>0:
                f1 = torch.nn.functional.normalize(torch.squeeze(self.expertBackbone(x1)))
                f2 = torch.nn.functional.normalize(torch.squeeze(self.expertBackbone(x2)))
                # f1 = torch.nn.functional.normalize(torch.squeeze(self.backbone(x1)))
                # f2 = torch.nn.functional.normalize(torch.squeeze(self.backbone(x2)))

            else:
                f1 = torch.squeeze(self.expertBackbone(x1))
                f2 = torch.squeeze(self.expertBackbone(x2))
                # f1 = torch.squeeze(self.backbone(x1))
                # f2 = torch.squeeze(self.backbone(x2))

            z1 = self.projector(f1)
            z2 = self.projector(f2)

            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(x1.shape[0])
            torch.distributed.all_reduce(c)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            # loss = self.scale_loss*(on_diag + self.lambd * off_diag)
            loss = (0.025 * (on_diag + 0.0051 * off_diag))

            # wandb.log({"loss expert": loss.item()})

        elif self.kd == 'L2' and self.retrospection:

            # print("--------I am doing retrospection and adaptation-------")
            f1 = torch.squeeze(self.backbone(x1))
            f2 = torch.squeeze(self.backbone(x2))

            f1Expert = torch.squeeze(self.copyExpert(x1)).detach()
            f2Expert = torch.squeeze(self.copyExpert(x2)).detach()

            # Map to old task
            z1 = self.projectorExp(f1)
            z1_2 = self.projectorExp(f2)

            # ======L2 + ssL2 norm
            lossAdapt = self.lambdaExp * (torch.dist(torch.nn.functional.normalize(z1), torch.nn.functional.normalize(f1Expert)) +
                                          torch.dist(torch.nn.functional.normalize(z1_2), torch.nn.functional.normalize(f2Expert)))

            # # decorrelative loss
            # lossAdapt = (barlow_loss_func(z1, f1Expert) + barlow_loss_func(z1_2, f2Expert)) / 2

            loss = lossAdapt

            if self.t > 0:

                if self.norm:
                    f1old = torch.nn.functional.normalize(torch.squeeze(self.oldBackbone(x1))).detach()
                    f2old = torch.nn.functional.normalize(torch.squeeze(self.oldBackbone(x2))).detach()
                else:
                    f1old = torch.squeeze(self.oldBackbone(x1)).detach()
                    f2old = torch.squeeze(self.oldBackbone(x2)).detach()

                # Map to old task
                z1 = self.projectorRet(f1)
                z1_2 = self.projectorRet(f2)

                # # ======L2 + ssL2 norm
                lossRetro = self.lambdapRet * (
                            torch.dist(torch.nn.functional.normalize(z1), torch.nn.functional.normalize(f1old)) +
                            torch.dist(torch.nn.functional.normalize(z1_2), torch.nn.functional.normalize(f2old)))

                # decorrelative loss
                # lossRetro = (barlow_loss_func(z1, f1old) + barlow_loss_func(z1_2, f2old)) / 2


                loss += lossRetro

                # wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": lossRetro.item()})
                wandb.log({ "loss lossAdapt": lossAdapt.item(), "loss Retrospection": lossRetro.item()})



            else:
                # wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": 0})
                wandb.log({"loss lossAdapt": lossAdapt.item(), "loss Retrospection": 0})

        elif self.kd == 'L2' and self.transfer:

            print("--------I am doing transfer-------")

            f1 = torch.squeeze(self.copyBackbone(x1)).detach()
            f2 = torch.squeeze(self.copyBackbone(x2)).detach()

            f1Expert = torch.squeeze(self.expertBackbone(x1))
            f2Expert = torch.squeeze(self.expertBackbone(x2))

            z1 = self.projectorExp(f1Expert)
            z1_2 = self.projectorExp(f2Expert)

            # ======L2
            loss = self.lambdaExp * (
                        torch.dist(torch.nn.functional.normalize(z1), torch.nn.functional.normalize(f1)) +
                        torch.dist(torch.nn.functional.normalize(z1_2), torch.nn.functional.normalize(f2)))

            wandb.log({"loss transfer": loss.item()})


        else:

            loss = 0

        return loss

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, concat=True):

        # gather keys before updating queue in distributed mode
        if concat:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity as in MoCo-v2

        # replace the keys at ptr (de-queue and en-queue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # move pointer
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def configure_optimizers(self, lrExp, mode):

        if mode == 0: # train

            params = [
                {'params': self.projector.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.expertBackbone.parameters(), 'lr':  lrExp *self.base_lr},
            ]
        elif(mode == 1): # retrospection
            params = [
                {'params': self.backbone.parameters(), 'lr': self.base_lr},
                {'params': self.projectorExp.parameters(), 'lr': self.base_lr},
                {'params': self.projectorRet.parameters(), 'lr': self.base_lr},
            ]

        elif(mode == 2): # transfer
            params = [
                {'params': self.expertBackbone.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.projectorExp.parameters(), 'lr': self.base_lr},
            ]

        optim = torch.optim.SGD(params, lr=self.base_lr, momentum=0.9, weight_decay=5e-4)
        max_steps = int(2.3 * 1500)#self.maxEpochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)

        if self.t<1 and mode == 0:
            return optim, None
        else:
            return optim, None




class vicReg(nn.Module):
    def __init__(self, projectorArc, batch_size, lambd, change_lr_scheduler, maxEpochs, diff_lr, kd_method, lambdapRet,
                 lambdaExp, lr, expProjSize, adaptSche, linearProj, linearProjRet, norm, mainArch,lambVect):
        super().__init__()
        ### Barlow Twins params ###
        self.projectorArc = projectorArc
        print("-----------------------------", self.projectorArc)
        self.batch_size = batch_size
        self.lambd = lambd
        self.scale_loss = 0.025
        self.linearProj = linearProj
        self.linearProjRet = linearProjRet
        self.norm = norm
        self.mainArch = mainArch
        self.lamb = lambVect

        ### Continual learning parameters ###
        self.kd = kd_method
        self.t = 0
        self.oldModel = None
        self.oldModelFull = None
        self._task_classifiers = None

        self.lambdapRet = lambdapRet
        self.lambdaExp = lambdaExp
        self.criterion = nn.CosineSimilarity(dim=1)
        self.retrospection = False
        self.transfer = False

        self.num_features = 512
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

        self.p2 = utilsProj.prediction_mlp(512, 256, self.linearProj)

        ### Training params
        self.change_lr_scheduler = change_lr_scheduler
        self.maxEpochs = maxEpochs
        self.diff_lr = diff_lr
        self.base_lr = lr  # 0.01
        self.lars_wrapper = False
        self.switchHead = True

        # log
        self.wandblog = False

        # Architecture
        if self.mainArch == "ResNet18":
            self.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.backbone.fc = nn.Identity()
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.backbone.maxpool = nn.Identity()
        elif self.mainArch == "ResNet18Com":
            self.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        elif self.mainArch == "ResNet9":
            self.backbone = ResNet9()

        elif self.mainArch == "SqueezeNet":
            self.backbone = SqueezeNet()

        # Distillation projectors
        self.expProjSize = expProjSize
        self.adaptSche = adaptSche
        self.adaptW = 1

        self.K = 65536
        self.register_buffer("queue", torch.randn(512, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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
        self.p2 = utilsProj.prediction_mlp(512, 256, self.linearProj)
        self.p2exp = utilsProj.prediction_mlp(512, 256, self.linearProj)
        self.bnExp = nn.BatchNorm1d(self.expProjSize, affine=False)
        self.bnRet = nn.BatchNorm1d(self.expProjSize, affine=False)
        self.projectorExp = utilsProj.prediction_mlp(self.expProjSize, 256, self.linearProj)
        self.projectorRet = utilsProj.prediction_mlp(self.expProjSize, int(self.expProjSize / 2), self.linearProjRet)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


    def forward(self, x1, x2):
        """Do SSL forward on input data stream"""

        if (not (self.retrospection) and not (self.tranfer)):

            if self.norm:
                f1 = torch.nn.functional.normalize(torch.squeeze(self.expertBackbone(x1)))
                f2 = torch.nn.functional.normalize(torch.squeeze(self.expertBackbone(x2)))

            else:
                f1 = torch.squeeze(self.expertBackbone(x1))
                f2 = torch.squeeze(self.expertBackbone(x2))

            x = self.projector(f1)
            y = self.projector(f2)

            repr_loss = F.mse_loss(x, y)

            x = torch.cat(FullGatherLayer.apply(x), dim=0)
            y = torch.cat(FullGatherLayer.apply(y), dim=0)
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            cov_x = (x.T @ x) / (self.batch_size - 1)
            cov_y = (y.T @ y) / (self.batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                self.num_features
            ) + off_diagonal(cov_y).pow_(2).sum().div(2048)

            loss = (
                    self.sim_coeff * repr_loss
                    + self.std_coeff * std_loss
                    + self.cov_coeff * cov_loss
            )
            return loss

            # wandb.log({"loss expert": loss.item()})

        elif self.kd == 'L2' and self.retrospection:
            """Perform adaptation and retrospection using L2 distance and distillation"""

            # print("--------I am doing retrospection and adaptation-------")
            f1 = torch.squeeze(self.backbone(x1))
            f2 = torch.squeeze(self.backbone(x2))

            f1Expert = torch.squeeze(self.copyExpert(x1)).detach()
            f2Expert = torch.squeeze(self.copyExpert(x2)).detach()

            # Map to old task
            z1 = self.projectorExp(f1)
            z1_2 = self.projectorExp(f2)

            # ======L2 + ssL2 norm
            lossAdapt = self.lambdaExp * (torch.dist(torch.nn.functional.normalize(z1), torch.nn.functional.normalize(f1Expert)) +
                                          torch.dist(torch.nn.functional.normalize(z1_2), torch.nn.functional.normalize(f2Expert)))


            loss = lossAdapt



            if self.t > 0:

                if self.norm:
                    f1old = torch.nn.functional.normalize(torch.squeeze(self.oldBackbone(x1))).detach()
                    f2old = torch.nn.functional.normalize(torch.squeeze(self.oldBackbone(x2))).detach()
                else:
                    f1old = torch.squeeze(self.oldBackbone(x1)).detach()
                    f2old = torch.squeeze(self.oldBackbone(x2)).detach()

                # Map to old task
                z1 = self.projectorRet(f1)
                z1_2 = self.projectorRet(f2)

                # ======L2 + ssL2 norm
                lossRetro = self.lambdapRet * (
                            torch.dist(torch.nn.functional.normalize(z1), torch.nn.functional.normalize(f1old)) +
                            torch.dist(torch.nn.functional.normalize(z1_2), torch.nn.functional.normalize(f2old)))


                loss += lossRetro

                wandb.log({ "loss lossAdapt": lossAdapt.item(), "loss Retrospection": lossRetro.item()})

            else:
                # wandb.log({"loss Adaptation": lossAdapt.item(), "loss Retrospection": 0})
                wandb.log({"loss lossAdapt": lossAdapt.item(), "loss Retrospection": 0})

        elif self.kd == 'L2' and self.transfer:

            print("--------I am doing transfer-------")

            f1 = torch.squeeze(self.copyBackbone(x1)).detach()
            f2 = torch.squeeze(self.copyBackbone(x2)).detach()

            f1Expert = torch.squeeze(self.expertBackbone(x1))
            f2Expert = torch.squeeze(self.expertBackbone(x2))

            z1 = self.projectorExp(f1Expert)
            z1_2 = self.projectorExp(f2Expert)

            # ======L2
            loss = self.lambdaExp * (
                        torch.dist(torch.nn.functional.normalize(z1), torch.nn.functional.normalize(f1)) +
                        torch.dist(torch.nn.functional.normalize(z1_2), torch.nn.functional.normalize(f2)))

            wandb.log({"loss transfer": loss.item()})


        else:

            loss = 0

        return loss

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, concat=True):

        # gather keys before updating queue in distributed mode
        if concat:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity as in MoCo-v2

        # replace the keys at ptr (de-queue and en-queue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # move pointer
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def configure_optimizers(self, lrExp, mode):

        if mode == 0: # train

            params = [
                {'params': self.projector.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.expertBackbone.parameters(), 'lr':  lrExp *self.base_lr},
            ]
        elif(mode == 1): # retrospection
            params = [
                {'params': self.backbone.parameters(), 'lr': self.base_lr},
                {'params': self.projectorExp.parameters(), 'lr': self.base_lr},
                {'params': self.projectorRet.parameters(), 'lr': self.base_lr},
            ]

        elif(mode == 2): # transfer
            params = [
                {'params': self.expertBackbone.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.projectorExp.parameters(), 'lr': self.base_lr},
            ]

        optim = torch.optim.SGD(params, lr=self.base_lr, momentum=0.9, weight_decay=5e-4)
        max_steps = int(2.3 * 1500)#self.maxEpochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)

        if self.t<1 and mode == 0:
            return optim, None
        else:
            return optim, None


# MoCo V2
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # CL parameters
        self.retrospection = False
        self.tranfer = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if (not (self.retrospection) and not (self.tranfer)):

            # compute query features
            q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            loss = self.criterion(logits, labels)

            return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        # self.conv1 = conv_block(3, 64)
        # self.conv2 = conv_block(64, 128, pool=True)
        # self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        #
        # self.conv3 = conv_block(128, 256, pool=True)
        # self.conv4 = conv_block(256, 512, pool=True)
        # self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        #
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())

        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.conv3 = conv_block(64, 128, pool=True)
        self.conv4 = conv_block(128, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=100):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        # self.conv10 = nn.Conv2d(512, class_num, 1)
        self.conv10 = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
        # self.conv10 = nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
        self.avg = nn.AdaptiveAvgPool2d(4)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)
        #
        #x = self.avg(f9).flat()
        # x = x.view(x.size(0), -1)

        return c10

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

def barlow_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3, scale_loss: float = 0.025
) -> torch.Tensor:
    """Computes Barlow Twins' loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
            Defaults to 5e-3.
        scale_loss (float, optional): final scaling factor of the loss. Defaults to 0.025.

    Returns:
        torch.Tensor: Barlow Twins' loss.
    """

    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N

    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = scale_loss * cdif.sum()
    return loss