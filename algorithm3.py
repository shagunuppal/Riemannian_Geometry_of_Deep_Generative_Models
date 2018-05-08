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
from PCA import *

from algorithm1 import *

model = VAE(784,400,20)
load_model()

T = 20
dt = 1.0 / T

def initial_velocity(z0):
	z = Variable(z0.view(20,1).data,requires_grad=True)
	sigma = torch.FloatTensor(np.identity(20))
	mu = torch.zeros(20,1)
	sigma_np = sigma.numpy()
	det = np.linalg.det(sigma_np)
	c = 1 /((math.pow(2 * math.pi, 10))*(math.sqrt(det))) 
	a = torch.exp((-1.0/2)*torch.mm(torch.t(z),z))
	latent_space = (a * c)
	latent_space.backward()
	k = z.grad.data.view(20)
	k_ = Variable(k, requires_grad = True)
	o = find_jacobian_1(model,k_)
	k1 = torch.mm(o,k.view(20,1))
	return k1	

def make_sigma(sig):
	sigma = torch.zeros(784,20)
	for i in range(20):
		sigma[i][i] = sig[i]
	return sigma

def compute_SVD(matrix):
	u, sigma, vh = torch.svd(matrix, some=False)
	return (u, sigma, vh)

def main3(z0, u0):
	x = []
	z = []
	u = []
	z.append(z0)
	u.append(u0)
	x.append(model.decode(z0))

	for i in range(0,T):
		xi = model.decode(z[len(z) - 1]).view(784)
		ui = u[len(u) - 1].view(784)
		xiplus1 = Variable(torch.add(xi.data, dt * ui).view(784), requires_grad=True)
		zxx = model.encode(xiplus1)
		ziplus1 = Variable(zxx[0].data, requires_grad=True)
		xiplus1 = model.decode(ziplus1)
		Jg = find_jacobian_1(model, ziplus1)
		U, sigma, vh = compute_SVD(Jg)
		sigma = make_sigma(sigma)
		U, sigma, vh, jgg = reduction(U, sigma, vh, Jg)
		uiplus1 = (torch.mm(torch.mm(U, U.t()),u[len(u) - 1])).view(784,1)
		#ui = torch.mm(torch.mm(U, U.t()),u[len(u) - 1].view(784,1))
		uiplus1 = (find_mod(u[len(u) - 1]) / find_mod(uiplus1)) * uiplus1
		u.append(uiplus1)
		z.append(ziplus1)
		x.append(xiplus1)
	make_image(z[len(z)-1].data.view(20),"algo3_final")
	make_image(z[0].data.view(20),"algo3_initial")
	return (z[len(z) - 1])

z0 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
u0 = initial_velocity(z0)
main3(z0,u0)
