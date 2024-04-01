import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class TensorDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, transform, offset=0, class_indices=None):
        self.labels = torch.LongTensor(data['y'])
        self.images = torch.FloatTensor(data['x'])
        self.transform = transform  # this is ignored
        self.offset = offset
        self.class_indices = class_indices

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.images[index]), self.labels[index]
        else:
            return self.images[index], self.labels[index]