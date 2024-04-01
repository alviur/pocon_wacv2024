import torch
import torch.nn as nn
import approach.utilsProj as utilsProj
import torchvision.models as models
# WandB Import the wandb library
import wandb

# Barlow twins
class BarlowTwins(nn.Module):
    def __init__(self, projectorArc, batch_size, lambd, change_lr_scheduler, maxEpochs, diff_lr, kd_method, lambdapRet,
                  lr, expProjSize, adaptSche, norm, mainArch):
        super().__init__()
        ### Barlow Twins params ###
        self.projectorArc = projectorArc
        print("-----------------------------", self.projectorArc)
        self.batch_size = batch_size
        self.lambd = lambd
        self.scale_loss = 0.025
        self.norm = norm
        self.mainArch = mainArch

        ### Continual learning parameters ###
        self.kd = kd_method
        self.t = 0
        self.oldModel = None
        self.oldModelFull = None
        self._task_classifiers = None

        self.lambdapRet = lambdapRet
        self.criterion = nn.CosineSimilarity(dim=1)



        ### Training params
        self.change_lr_scheduler = change_lr_scheduler
        self.maxEpochs = maxEpochs
        self.diff_lr = diff_lr
        self.base_lr = lr  # 0.01
        self.lars_wrapper = False

        # log
        self.wandblog = False

        # Architecture
        if self.mainArch == "ResNet18":
            self.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.backbone.fc = nn.Identity()
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.backbone.maxpool = nn.Identity()
            dim_out = 512
        elif self.mainArch == "ResNet18Com":
            self.backbone = models.resnet18(pretrained=False, zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        elif self.mainArch == "ResNet9":
            self.backbone = ResNet9()
            dim_out = 512

        elif self.mainArch == 'shufflenet':
            from torchvision.models import shufflenet_v2_x0_5
            self.backbone = shufflenet_v2_x0_5(pretrained=False)
            self.backbone.fc = nn.Identity()  # nn.Linear(1024, 512)
            self.backbone.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
            dim_out = 1024

        self.p2 = _prediction_mlp(dim_out, 256)


        # Distillation projectors
        self.expProjSize = expProjSize
        self.adaptSche = adaptSche
        self.adaptW = 1

        # projector
        sizes = [dim_out] + list(map(int, self.projectorArc.split('_')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x1, x2):

        if self.norm:
            f1 = torch.nn.functional.normalize(torch.squeeze(self.backbone(x1)))
            f2 = torch.nn.functional.normalize(torch.squeeze(self.backbone(x2)))

        else:
            f1 = torch.squeeze(self.backbone(x1))
            f2 = torch.squeeze(self.backbone(x2))

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


        if self.kd == 'pfr' and self.t > 0:


            f1Old = torch.squeeze(self.oldBackbone(x1))
            f2Old = torch.squeeze(self.oldBackbone(x2))

            p2_1 = self.p2(f1)
            p2_2 = self.p2(f2)

            lossKD = self.lambdapRet * (-(self.criterion(p2_1, f1Old.detach()).mean()
                                        + self.criterion(p2_2, f2Old.detach()).mean()) * 0.5)

            wandb.log({"loss BT": loss.item(),"loss Retrospection": lossKD.item()})

            loss += lossKD

        else:
            wandb.log({"loss BT": loss.item()})



        return loss


    def configure_optimizers(self):

        if self.t < 1:
            params = list(self.backbone.parameters())
            params += list(self.projector.parameters())

            optim = torch.optim.SGD(params, lr=self.base_lr, momentum=0.9, weight_decay=5e-4)
            max_steps = (int(2.3 * self.maxEpochs)) if self.change_lr_scheduler else self.maxEpochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)
            self.scheduler = scheduler
            return optim, scheduler

        else:
            lr = self.base_lr
            params = [
                {'params': self.backbone.parameters(), 'lr': lr*0.01 if self.diff_lr else 0.5*0.8*lr},
                {'params': self.projector.parameters(), 'lr': lr*0.3 if self.diff_lr else 0.8*lr},
            ]
            if self.kd == 'pfr':
                params.append({'params': self.p2.parameters(), 'lr': 0.8*lr})

            print('Optimizer lr: ', [d['lr'] for d in params])
            optim = torch.optim.SGD(params, lr=lr * 0.8, momentum=0.9, weight_decay=5e-4)
            return optim, None

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

def _prediction_mlp(in_dims: int, h_dims: int):
    prediction = nn.Sequential(nn.Linear(in_dims, h_dims, bias=False),
                               nn.BatchNorm1d(h_dims),
                               nn.ReLU(inplace=True),  # hidden layer
                               nn.Linear(h_dims, in_dims))  # output layer

    return prediction


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

