
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
    x = x.data
    x1 = x.numpy()
    for i in range(20):
        q = x1[i]
        p += q*q
    return math.sqrt(p)

def linear_distance(z1,z2):
    x = z2 - z1
    return find_mod1(x)

# def arc_length(model, z1, z2):
#     x = 0 
#     x = model.decode(z2) - model.decode(z1)
#     return x * T

# def geodesic_length(model, z_collection):
#     x = 0
#     for i in range(1,T):
#         x += model.decode(z_collection[i]) -  model.decode(z_collection[i-1])
#     return x * T