import torch
import torch.nn as nn
import torch.nn.functional as F

def prediction_mlp(in_dims, h_dims, linear):

    if linear==0:

        prediction = nn.Sequential(nn.Linear(in_dims, in_dims, bias=True))

    elif(linear==1):
        prediction = nn.Sequential(nn.Linear(in_dims, h_dims, bias=False),
                                   nn.BatchNorm1d(h_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(h_dims, in_dims))  # output layer

    elif (linear == 2):
        prediction = nn.Sequential(nn.Linear(in_dims, h_dims, bias=False),
                                   nn.BatchNorm1d(h_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(h_dims, h_dims),
                                   nn.BatchNorm1d(h_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(h_dims, in_dims)
                                   )  # output layer

    elif (linear == 3):# following https://github.com/UCDvision/simreg
        prediction = nn.Sequential(nn.Linear(in_dims, 2*in_dims, bias=False),
                                   nn.BatchNorm1d(2*in_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(2*in_dims, in_dims),
                                   nn.BatchNorm1d(in_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(in_dims, 2*in_dims),
                                   nn.BatchNorm1d(2*in_dims),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(2*in_dims, in_dims)
                                   )  # output layer

    elif (linear == 4):# following https://github.com/UCDvision/simreg
        prediction = nn.Sequential(nn.Linear(in_dims, 2 * in_dims, bias=False),
                                   nn.BatchNorm1d(2 * in_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(2 * in_dims, in_dims),
                                   nn.BatchNorm1d(in_dims),
                                   nn.ReLU(inplace=True),  # hidden layer
                                   nn.Linear(in_dims, 2 * in_dims),
                                   nn.BatchNorm1d(2 * in_dims),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(2 * in_dims, in_dims),
                                   nn.BatchNorm1d(in_dims),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_dims, 2 * in_dims),
                                   nn.BatchNorm1d(2 * in_dims),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(2 * in_dims, in_dims)
                                   )  # output layer

    return prediction

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


def alignLoss(feat_s, feat_t):

    feat_s = F.normalize(feat_s, dim=-1, p=1)
    feat_t = F.normalize(feat_t, dim=-1, p=1)
    return 2 - 2 * (feat_s * feat_t).sum(dim=-1).mean()

def soft_cross_entropy(student_logit, teacher_logit):
    '''
    :param student_logit: logit of the student arch (without softmax norm)
    :param teacher_logit: logit of the teacher arch (already softmax norm)
    :return: CE loss value.
    '''
    return -(teacher_logit * torch.nn.functional.log_softmax(student_logit, 1)).sum()/student_logit.shape[0]

def simMatrixSEED(s_emb,t_emb,queue):

    """
    Input:
        t_emb: teacher features
        s_emb: student features
    Output:
        student logits, teacher logits
    """

    # compute query features
    s_emb = nn.functional.normalize(s_emb, dim=1)

    # compute key features
    with torch.no_grad():  # no gradient to keys
        t_emb = nn.functional.normalize(t_emb, dim=1)

    # cross-Entropy Loss
    logit_stu = torch.einsum('nc,ck->nk', [s_emb, queue.clone().detach()])
    logit_tea = torch.einsum('nc,ck->nk', [t_emb, queue.clone().detach()])

    logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)
    logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)

    logit_stu = torch.cat([logit_s_p, logit_stu], dim=1)
    logit_tea = torch.cat([logit_t_p, logit_tea], dim=1)

    # compute soft labels
    logit_stu /= 0.07#self.t

    logit_tea = nn.functional.softmax(logit_tea / 1e-4, dim=1)



    return soft_cross_entropy(logit_stu, logit_tea)

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
    for name, param in model.state_dict().items():
        if (name not in stateDictSaved):
            continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        #     param = param.data
        if (name[0:8]=='backbone') or (name[0:14]=='expertBackbone') or (name[0:13]=='currentExpert'):
            print("loading", name)
            model.state_dict()[name].copy_(stateDictSaved[name])

    return model

def load_backboneProj(model, stateDictSaved):
    # own_state = model.state_dict()
    for name, param in model.state_dict().items():
        if (name not in stateDictSaved):
            continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        #     param = param.data
        if (name[0:8]=='backbone') or (name[0:14]=='expertBackbone') or (name[0:13]=='currentExpert') or (name[0:12]=='projectorRet'):
            print("loading", name)
            model.state_dict()[name].copy_(stateDictSaved[name])
        # else:
        #     print("Not", name)


    return model


def load_my_state_dictExp(model, stateDictSaved):
    # own_state = model.state_dict()
    for name, param in model.state_dict().items():
        print(name[15:], name)
        if name not in stateDictSaved:
            continue
        # if isinstance(param, Parameter):
        #     # backwards compatibility for serialized parameters
        #     param = param.data
        print("loading", name, name[15:])
        model.state_dict()[name].copy_(stateDictSaved[name])

    return model

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

# for p1, p2 in zip(self.modelFB.expertBackbone.parameters(), self.modelFB.backbone.parameters()):
#     if p1.data.ne(p2.data).sum() > 0:
#         print("DIFERENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

