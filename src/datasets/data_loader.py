import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100
from torchvision.datasets import CIFAR10 as TorchVisionCIFAR10
from torchvision.datasets import SVHN as TorchVisionSVHN
from torchvision.datasets import MNIST as TorchVisionMNIST

from . import base_dataset as basedat
from . import memory_dataset as memd
from . import tensor_dataset as tensd
from .dataset_config import dataset_config


def get_loaders(datasets, num_tasks, nc_first_task, batch_size, num_workers, pin_memory, validation=.1):
    nc_first_task = None if (nc_first_task is not None and nc_first_task <= 0) else nc_first_task
    trn_load = []
    val_load = []
    tst_load = []
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      tensorFlag=False,
                                                      normalize=dc['normalize']) if cur_dataset != 'mnist1d' else (
        None, None)

        # datasets
        trn_dset, val_dset, tst_dset, curtaskcla = get_datasets(cur_dataset, dc['path'], num_tasks, nc_first_task,
                                                                validation=validation,
                                                                trn_transform=trn_transform,
                                                                tst_transform=tst_transform,
                                                                class_order=dc['class_order'])



        # Apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # Reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]

        # Extend real taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))
    return trn_load, val_load, tst_load, taskcla


def get_datasets(dataset, path, num_tasks, nc_first_task, validation, trn_transform, tst_transform, class_order=None,
                 shuffle_classes=True):
    trn_dset = []
    val_dset = []
    tst_dset = []

    # Base Dataset style datasets
    if dataset in ['flowers', 'scenes', 'birds', 'cars', 'aircraft', 'actions', 'letters', 'tiny_imagenet',
                   'vggface2', 'imagenet_256', 'imagenet_256_noTrans', 'imagenet100'] or dataset.startswith('imagenet_subset'):
        # read data paths and compute splits
        all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                            validation=validation, shuffle_classes=class_order is None,
                                                            class_order=class_order)
        # set dataset type
        Dataset = basedat.BaseDataset

    elif dataset == 'imagenet_no_birds':
        # read data paths and compute splits
        raise NotImplementedError(
            'imagenet_no_birds dataset is missing the implementation of number of classes for first task')
        all_data, taskcla = basedat.get_data_imagenet_no_birds(path, num_tasks=num_tasks, validation=validation)
        # set dataset type
        Dataset = basedat.BaseDataset

    elif 'mnist' == dataset:
        tvmnist_trn = TorchVisionMNIST(path, train=True, download=True)
        tvmnist_tst = TorchVisionMNIST(path, train=False, download=True)
        trn_data = {'x': tvmnist_trn.data.numpy(), 'y': tvmnist_trn.targets.tolist()}
        tst_data = {'x': tvmnist_tst.data.numpy(), 'y': tvmnist_tst.targets.tolist()}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         validation=validation, shuffle_classes=class_order is None,
                                                         class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'cifar10' == dataset:
        tvcifar_trn = TorchVisionCIFAR10(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR10(path, train=False, download=True)
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         validation=validation, shuffle_classes=class_order is None,
                                                         class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif 'cifar100' in dataset:
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True)
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True)
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         validation=validation, shuffle_classes=class_order is None,
                                                         class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'svhn' or dataset == 'svhn_noTrans':
        tvsvhn_trn = TorchVisionSVHN(path, split='train', download=True)
        tvsvhn_tst = TorchVisionSVHN(path, split='test', download=True)
        trn_data = {'x': tvsvhn_trn.data.transpose(0, 2, 3, 1), 'y': tvsvhn_trn.labels}
        tst_data = {'x': tvsvhn_tst.data.transpose(0, 2, 3, 1), 'y': tvsvhn_tst.labels}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         validation=validation, shuffle_classes=shuffle_classes,
                                                         class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset == 'mnist1d':
        if not os.path.exists(f'{path}/mnist1d_data.pkl'):
            print('Downloading MNIST1D dataset...')
            import requests
            url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
            r = requests.get(url, allow_redirects=True)
            open(f'{path}/mnist1d_data.pkl', 'wb').write(r.content)

        import pickle
        with open(f'{path}/mnist1d_data.pkl', 'rb') as handle:
            data = pickle.load(handle)

        trn_data = {'x': data['x'].astype(np.float32), 'y': data['y']}
        tst_data = {'x': data['x_test'].astype(np.float32), 'y': data['y_test']}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         validation=validation, shuffle_classes=class_order is None,
                                                         class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset
        # Dataset = tensd.TensorDataset

    elif 'imagenet_32' in dataset:
        import pickle
        # Load data
        x_trn, y_trn = [], []
        for i in range(1, 11):
            with open(os.path.join(path, 'train_data_batch_{}'.format(i)), 'rb') as f:
                d = pickle.load(f)
            x_trn.append(d['data'])
            y_trn.append(np.array(d['labels']) - 1)  # Labels from 0 to 999
        with open(os.path.join(path, 'val_data'), 'rb') as f:
            d = pickle.load(f)
        x_trn.append(d['data'])
        y_tst = np.array(d['labels']) - 1  # Labels from 0 to 999
        # Reshape data
        for i, d in enumerate(x_trn, 0):
            x_trn[i] = d.reshape(d.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        x_tst = x_trn[-1]
        x_trn = np.vstack(x_trn[:-1])
        y_trn = np.concatenate(y_trn)
        trn_data = {'x': x_trn, 'y': y_trn}
        tst_data = {'x': x_tst, 'y': y_tst}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data,
                                                         num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                         validation=validation, shuffle_classes=shuffle_classes,
                                                         class_order=class_order)
        # set dataset type
        Dataset = memd.MemoryDataset

    elif dataset in ['exp4_1-1', 'exp4_2-1', 'exp4_3-1', 'exp4_4-1', 'exp4_1-2', 'exp4_2-2', 'exp4_3-2', 'exp4_4-2',
                     'exp4_1-3', 'exp4_2-3', 'exp4_3-3', 'exp4_4-3', 'exp4_1-4', 'exp4_2-4', 'exp4_3-4', 'exp4_4-4']:
        # read data paths and compute splits
        all_data, taskcla, class_indices = basedat.get_data(path, num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                            validation=validation, shuffle_classes=shuffle_classes,
                                                            class_order=class_order)
        # set dataset type
        Dataset = basedat.BaseDataset
    else:
        raise RuntimeError(f'Bad dataset: {dataset}')

    # get datasets
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, offset, class_indices))
        val_dset.append(Dataset(all_data[task]['val'], tst_transform, offset, class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, offset, class_indices))
        offset += taskcla[task][1]

    return trn_dset, val_dset, tst_dset, taskcla


def get_transforms(resize, pad, crop, flip, normalize, tensorFlag=True):
    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    if tensorFlag:
        trn_transform_list.append(transforms.ToTensor())
        tst_transform_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)
