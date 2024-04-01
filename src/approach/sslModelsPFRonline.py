import torch
import torch.nn as nn
import approach.utilsProj as utilsProj
import torchvision.models as models
from copy import deepcopy
from datasets.exemplars_selection import override_dataset_transform
from torch.utils.data.dataloader import DataLoader
import time
import numpy as np
from loggers.exp_logger import ExperimentLogger
import torch.nn.functional as F
# WandB Import the wandb library
import wandb

# Barlow twins
class BarlowTwins(nn.Module):
    def __init__(self, projectorArc, batch_size, lambd, change_lr_scheduler, maxEpochs, diff_lr, kd_method, lambdapRet,
                 lambdaExp, lr, expProjSize, adaptSche, linearProj, linearProjRet, norm, mainArch,omni_trn_loader,
                 omni_val_loader,val_transforms,omni_tst_loader,test_every_n_steps,output_path):
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
        self.test_every_n_steps = test_every_n_steps
        self.output_path = output_path
        self.omni_trn_loader = omni_trn_loader
        self.omni_val_loader = omni_val_loader
        self.val_transforms = val_transforms
        self.omni_tst_loader = omni_tst_loader
        self._task_classifiers = []

        self.trainX = {}
        self.trainXexp = {}
        self.trainY = {}
        self.valX = {}
        self.valXexp = {}
        self.valY = {}
        self.head_classifier_lr = 5e-3
        self.lr_patience = 20
        self.classifier_nepochs = 200
        self.lr_factor = 3
        self.lr_min = 5e-7

        ### Continual learning parameters ###
        self.kd = kd_method
        self.t = 0
        self.oldModel = None
        self.oldModelFull = None
        self.lamb = None
        self.lambdapRet = lambdapRet
        self.lambdaExp = lambdaExp
        self.criterion = nn.CosineSimilarity(dim=1)
        self.retrospection = False
        self.transfer = False
        self.train_history = []
        self.expertAccu = []

        self.num_features = 2048
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0
        self._encoder_emb_dim = 512

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
        elif self.mainArch == "ResNet9":
            self.backbone = ResNet9()

        elif self.mainArch == "SqueezeNet":
            self.backbone = SqueezeNet()

        self.oldBackbone = None

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

    def forward(self, x1, x2, batch_idx):

        if (1):

            if (batch_idx % self.test_every_n_steps) == 0 and batch_idx > 0:
                print('Validation...')
                acc = self.eval_omni(batch_idx)
                self.train_history.append(acc)
                print('Train history: ')
                print(self.train_history)

            if (self.kd == 'ft'):
                f1 = torch.squeeze(self.backbone(x1))
                f2 = torch.squeeze(self.backbone(x2))

            elif self.norm:
                f1 = torch.nn.functional.normalize(torch.squeeze(self.expertBackbone(x1)))
                f2 = torch.nn.functional.normalize(torch.squeeze(self.expertBackbone(x2)))

            else:
                f1 = torch.squeeze(self.expertBackbone(x1))
                f2 = torch.squeeze(self.expertBackbone(x2))


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

            wandb.log({"loss expert": loss.item()})

            if self.kd == 'pfr' and self.oldBackbone != None:
                f1Old = torch.squeeze(self.oldBackbone(x1))
                f2Old = torch.squeeze(self.oldBackbone(x2))

                p2_1 = self.p2(f1)
                p2_2 = self.p2(f2)

                lossKD = self.lambdapRet * (-(self.criterion(p2_1, f1Old.detach()).mean()
                                              + self.criterion(p2_2, f2Old.detach()).mean()) * 0.5)

                wandb.log({"loss BT": loss.item(), "loss Retrospection": lossKD.item()})

                loss += lossKD
        else:

            loss = 0

        return loss

    def eval_omni(self, step_num):
        # train and eval - like task '0', then offset is also '0'
        omni_head = self._train_classifier(0, self.omni_trn_loader[0], self.omni_val_loader[0],
                                           f"omni-classifier-iteration-{step_num}")
        test_loss, omni_head_acc_taw, omni_head_acc_tag = self.eval(
            0, self.omni_tst_loader[0], heads_to_evaluate=[omni_head]
        )
        print(f"Omni-head TAG ACC: {omni_head_acc_tag}")
        wandb.log({"Omni-head TAG ACC:": omni_head_acc_tag})

        models_to_save = {
            'simsiam': self.state_dict()
        }
        torch.save(models_to_save, self.output_path+'/model-task_'+str(0)+'.ckpt')
        print("Saved in: ", self.output_path)


        return omni_head_acc_tag

    def eval(self, t, orig_val_loader, heads_to_evaluate=None):
        with override_dataset_transform(orig_val_loader.dataset, self.val_transforms) as _ds_val:  # no data aug
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
                modelT = deepcopy(self.backbone).cuda()
                # modelTexp = deepcopy(self.modelFB.expertBackbone).to(self.device)
                # modelT = deepcopy(self.modelFB.expertBackbone).to(self.device)
                modelTexp = deepcopy(self.expertBackbone).cuda()
                modelTProj = deepcopy(self.projector).cuda()

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
                    r1 = torch.nn.functional.normalize(modelT(img_1.cuda()).flatten(start_dim=1))
                    r1exp = modelTexp(img_1.cuda()).flatten(start_dim=1)
                    z1 = modelTProj(r1)

                    all_z1.append(z1)
                    all_f1.append(r1)


                    loss = 0.0  # self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2
                    heads = heads_to_evaluate if heads_to_evaluate else self._task_classifiers
                    headsexp = [self.expertClass]
                    outputs = [h(r1.cuda()) for h in heads]
                    outputsexp = [h(r1exp.cuda()) for h in headsexp]
                    # outputsexp = self.expertClass(r1exp.to(self.device))
                    single_task = (heads_to_evaluate is not None) and (len(heads_to_evaluate) == 1)
                    single_taskexp = (heads_to_evaluate is not None) and (len(heads_to_evaluate) == 1)

                    pred = torch.cat(outputs, dim=1).argmax(1)
                    hits_taw = (pred == targets.cuda()).float()
                    hits_tag = hits_taw

                    # import pdb;
                    # pdb.set_trace()

                    pred = torch.cat(outputsexp, dim=1).argmax(1)
                    hits_tawexp = (pred == targets.cuda()).float()
                    hits_tagexp = hits_tawexp
                    # hits_tawexp, hits_tagexp = 0,0
                    # Log
                    total_loss += loss * len(targets)  # TODO
                    total_acc_taw += hits_taw.sum().cpu().item()
                    total_acc_tag += hits_tag.sum().cpu().item()
                    total_acc_tawexp += hits_tawexp.sum().cpu().item()
                    total_acc_tagexp += hits_tagexp.sum().cpu().item()
                    # total_acc_tawexp = 0
                    # total_acc_tagexp = 0
                    total_num += len(targets)

            all_z1 = torch.cat(all_z1)
            all_f1 = torch.cat(all_f1)
            if len(all_p1) > 0:
                all_p1 = torch.cat(all_p1)

        # wandb.log({"omni-head acc": 100 * (total_acc_tag / total_num)})

        self.expertAccu.append(total_acc_tawexp / total_num)
        wandb.log({"Omnihead Expert": total_acc_tawexp / total_num, 'Task': t + 1})
        print("---------Expert test accu: ", self.expertAccu)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def _train_Expert(self, t, trn_loader, val_loader, name='classifier'):

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

        _task_classifier.cuda()
        _task_classifierexp.cuda()
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
                _x = self.trainXexp[str(t)][index:index + trn_loader.batch_size, :]
                y = self.trainY[str(t)][index:index + trn_loader.batch_size]
                _x = _x.detach()
                # forward pass
                mlp_preds = _task_classifier(_x.cuda())
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
                    _x = self.valXexp[str(t)][index:index + val_loader.batch_size, :]
                    _xexp = self.valXexp[str(t)][index:index + val_loader.batch_size, :]
                    y = self.valY[str(t)][index:index + val_loader.batch_size]
                    _x = _x.detach();
                    _xexp = _x.detach()
                    # forward pass
                    mlp_preds = _task_classifier(_x.cuda())
                    mlp_loss = F.cross_entropy(mlp_preds, y)
                    val_loss += mlp_loss.item()
                    n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                    n_all = y.size()[0]
                    _val_acc = n_corr / n_all
                    # print(f"{self.name} online acc: {train_acc}")
                    acc_correct += n_corr
                    acc_all += n_all

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
                best_model = deepcopy(_task_classifier.model.state_dict())
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

            print()

        time_taken = time.time() - clock0
        _task_classifier.model.load_state_dict(best_model)
        _task_classifier.eval()
        print(f'{name} - Best ACC: {100 * best_val_acc:.1f} time taken: {time_taken:5.1}s')

        return _task_classifier

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

        _task_classifier.cuda()
        _task_classifierexp.cuda()
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
                mlp_preds = _task_classifier(_x.cuda())
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
                    mlp_preds = _task_classifier(_x.cuda())
                    mlp_loss = F.cross_entropy(mlp_preds, y)
                    val_loss += mlp_loss.item()
                    n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                    n_all = y.size()[0]
                    _val_acc = n_corr / n_all
                    # print(f"{self.name} online acc: {train_acc}")
                    acc_correct += n_corr
                    acc_all += n_all

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
                best_model = deepcopy(_task_classifier.model.state_dict())
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

            print()

        time_taken = time.time() - clock0
        _task_classifier.model.load_state_dict(best_model)
        _task_classifier.eval()
        print(f'{name} - Best ACC: {100 * best_val_acc:.1f} time taken: {time_taken:5.1}s')

        self.expertClass = self._train_Expert(t, trn_loader, val_loader, name='classifier')

        return _task_classifier

    def get_embeddings(self, t, trn_loader, val_loader):
        # Get backbone
        modelT = deepcopy(self.backbone).cuda()
        # modelT = deepcopy(self.modelFB.expertBackbone).to(self.device)
        modelTexpert = deepcopy(self.expertBackbone).cuda()
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

        trainX = torch.zeros((batchFloorT, self._encoder_emb_dim), dtype=torch.float).cuda()
        trainY = torch.zeros(batchFloorT, dtype=torch.long).cuda()
        valX = torch.zeros((batchFloorV, self._encoder_emb_dim), dtype=torch.float).cuda()
        valY = torch.zeros(batchFloorV, dtype=torch.long).cuda()

        trainXexp = torch.zeros((batchFloorT, self._encoder_emb_dim), dtype=torch.float).cuda()
        valXexp = torch.zeros((batchFloorV, self._encoder_emb_dim), dtype=torch.float).cuda()

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
                _x = modelT(img_1.cuda()).flatten(start_dim=1)
                _xexp = modelTexpert(img_1.cuda()).flatten(start_dim=1)
                _x = _x.detach();
                _xexp = _xexp.detach()
                y = torch.LongTensor((y).long().cpu()).cuda()
                trainX[contBatch:contBatch + trn_loader.batch_size, :] = _x
                trainY[contBatch:contBatch + trn_loader.batch_size] = y
                trainXexp[contBatch:contBatch + trn_loader.batch_size, :] = _xexp
                contBatch += trn_loader.batch_size

            contBatch = 0
            for img_1, y in _val_loader:
                _x = modelT(img_1.cuda()).flatten(start_dim=1)
                _xexp = modelTexpert(img_1.cuda()).flatten(start_dim=1)
                _x = _x.detach();
                _xexp = _xexp.detach()
                y = torch.LongTensor((y).long().cpu()).cuda()
                valX[contBatch:contBatch + _val_loader.batch_size, :] = _x
                valY[contBatch:contBatch + _val_loader.batch_size] = y
                valXexp[contBatch:contBatch + _val_loader.batch_size, :] = _xexp
                contBatch += _val_loader.batch_size

        return torch.nn.functional.normalize(trainX), trainY, torch.nn.functional.normalize(valX), valY, trainXexp, valXexp

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
                {'params': self.projector.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.expertBackbone.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.backbone.parameters(), 'lr': self.base_lr},
                {'params': self.projectorExp.parameters(), 'lr': self.base_lr},
                {'params': self.projectorRet.parameters(), 'lr': self.base_lr},
            ]

        elif(mode == 2): # transfer
            params = [
                {'params': self.expertBackbone.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.projector.parameters(), 'lr': lrExp * self.base_lr},
                {'params': self.projectorExp.parameters(), 'lr': self.base_lr},
            ]

        # if mode == 0: # train
        #
        #     params = [
        #         {'params': self.projector.parameters(), 'lr': lrExp * self.base_lr},
        #         {'params': self.expertBackbone.parameters(), 'lr': lrExp * self.base_lr},
        #         {'params': self.backbone.parameters(), 'lr': self.base_lr},
        #         {'params': self.projectorExp.parameters(), 'lr': self.base_lr},
        #         {'params': self.projectorRet.parameters(), 'lr': self.base_lr},
        #     ]

        optim = torch.optim.SGD(params, lr=self.base_lr, momentum=0.9, weight_decay=5e-4)
        max_steps =  self.maxEpochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_steps)
        if self.change_lr_scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = None


        # if self.lars_wrapper:
        #     optim = LARSWrapper(
        #         optim,
        #         eta=0.02,  # trust coefficient
        #         clip=True
        #     )

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