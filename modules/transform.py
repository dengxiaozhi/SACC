import torchvision
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from modules.RandAugment import RandAugment
import random
from PIL import ImageFilter


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class GaussianBlur2(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Transforms:
    def __init__(self, size, s=1.0, mean=None, std=None, blur=False):
        self.weak_transform = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.weak_transform.append(GaussianBlur(kernel_size=23))
        self.weak_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor(),
        ]
        if mean and std:
            self.weak_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.weak_transform = torchvision.transforms.Compose(self.weak_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur2([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=5, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        return self.weak_transform(x), self.weak_transform(x), self.strong_transform(x)
