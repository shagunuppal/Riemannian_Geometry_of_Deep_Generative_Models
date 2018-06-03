
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
from torchvision.datasets import MNIST
from torch.autograd.gradcheck import zero_gradients
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import math

def find_mod1(x):
    # x is float tensor
    p = 0
    x = x.data.view(20)
    x1 = x.numpy()
    for i in range(20):
        q = x1[i]
        p += q*q
    return math.sqrt(p)

def linear_distance(z1,z2):
    x = z2 - z1
    return find_mod1(x)
