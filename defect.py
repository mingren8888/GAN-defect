import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FL
from PIL import Image, ImageDraw
import numpy as np


class DefectAdder(object):
    def __init__(self, mode='geometry', resize=True, size_range=(0.05, 0.6), defect_shape=('circle', 'square')):
        self.mode = mode
        self.resize = resize
        self.size_range = size_range
        self.defect_shape = defect_shape

    def __call__(self, input):
        # assert isinstance(input, Image)
        # assert len(input.shape) == 3
        output = self.add_defect(input)
        return [input, output]

    def add_defect(self, input):
        w, h = input.size
        draw = ImageDraw.Draw(input)
        shape = np.random.choice(self.defect_shape)
        size_ratio = np.random.uniform(self.size_range[0], self.size_range[1])
        if shape == 'circle':
            x = int(np.random.random() * w)
            y = int(np.random.random() * h)
            size = int(size_ratio * min(w, h) * 0.5)
            color = tuple(np.random.randint(0, 255, 3))

            draw.ellipse([x, y, x + size, y + size], fill=color)
        elif shape == 'square':
            x = int(np.random.random() * w)
            y = int(np.random.random() * h)
            size = int(size_ratio * min(w, h) * 0.5)
            color = tuple(np.random.randint(0, 255, 3))
            draw.rectangle([x, y, x + size, y + size], fill=color)

        return input

    def __repr__(self):
        return self.__class__.__name__ + 'mode={}'.format(self.mode)


class NormalizeList(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensors: Normalized Tensor image.
        """
        for i in range(len(tensors)):
            tensors[i] = FL.normalize(tensors[i], self.mean, self.std, self.inplace)
        return tensors

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensorList(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pics):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for i in range(len(pics)):
            pics[i] = FL.to_tensor(pics[i])
        return pics

    def __repr__(self):
        return self.__class__.__name__ + '()'
