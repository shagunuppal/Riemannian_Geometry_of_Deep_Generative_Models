# Algorithm 2 : Parallel Translation

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

from algorithm1 import *

model = VAE(784,400,20)
load_model()

def compute_SVD(matrix):
	u, sigma, vh = np.linalg.svd(a, full_matrices=True)
	return (u, sigma, vh)

def main2(z_collection, v0):
	u = []
	u0 = torch.matmul(find_jacobian_1(model, z_collection[1]), v0)
	u.append(u0)
	T  = len(z_collection) - 1
	
	for i in range (T):
		xi = model.decode(z_collection[i])
		u, sigma, vh = compute_SVD(xi)
		ui = np.dot(u.transpose(), u, u[len(u) - 1])
		ui = (find_mod(u[len(u) - 1]) / find_mod(ui)) * ui
		u.append(ui)

	ut = u[len(u) - 1]
	vt_ = find_jacobian(model.decode(z_collection[len(z_collection) - 1]))
	vt = np.matmul(vt_, ut)
	return vts






