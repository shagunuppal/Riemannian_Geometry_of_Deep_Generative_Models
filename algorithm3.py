# Algorithm 3 : Geodesic Shooting

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
from algorithm2 import *

model = VAE(784,400,20)
load_model()

T = 10
dt = 1 / T

def main3(z0, u0):
	x = []
	z = []
	u = []
	z.append(z0)
	u.append(u0)
	x.append(model.decode(z0))

	for i in range(0,T):
		xi = model.decode(z[len(z) - 1])
		ui = u[len(u) - 1]
		xiplus1 = xi + dt * ui
		ziplus1 = model.encode(xiplus1)
		xiplus1 = model.decode(ziplus1)
		Jg = find_jacobian_1(ziplus1)
		u, sigma, vh = compute_SVD(Jg)
		uiplus1 = np.dot(u.transpose(), u, ui)
		uiplus1 = (find_mod(ui) / find_mod(uiplus1)) * uiplus1

		u.append(uiplus1)
		z.append(ziplus1)
		x.append(xiplus1)

	return (z[len(z) - 1])



