import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np


class DefectAdder(object):
    def __init__(self, mode='geometry', resize=True, size_range=(0.05, 0.6), defect_shape=('circle', 'square')):
        self.mode = mode
        self.resize = resize
        self.size_range = size_range
        self.defect_shape = defect_shape

    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            input = input.numpy()
        assert len(input.shape) == 4
        result = []
        for b in range(input.shape[0]):
            img = input[b, :, :, :]
            img = self.add_defect(img)
            result.append(img)
        output = np.concatenate(result)
        return output

    def add_defect(self, input):
        # TODO: add defect using pillow
        assert isinstance(input, np.ndarray)
        c, w, h = input.shape

        shape = np.random.choice(self.defect_shape)
        size_ratio = np.random.uniform(self.size_range[0], self.size_range[1])
        if shape == 'circle':
            x = int(np.random.random() * w)
            y = int(np.random.random() * h)
            size = int(size_ratio * min(w, h) * 0.5)
            color = np.random.random(3) * 255
        elif shape == 'square':
            x = int(np.random.random() * w)
            y = int(np.random.random() * h)
            size = int(size_ratio * min(w, h) * 1)
            color = np.random.random(3) * 255

        return input


