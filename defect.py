import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np


class DefectAdder(object):
    def __init__(self, mode='geometry', resize=True, size_range=(0.1, 0.6)):
        self.mode = mode
        self.resize = resize
        self.size_range = size_range

    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            input = input.numpy()

    def add_defect(self, input):
        assert isinstance(input, np.ndarray)
