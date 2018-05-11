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

from algorithm1_new import *

model = VAE(784,450,200,20)
load_model()

T = 4
dt = 1.0 / T

def initial_velocity(z):
	b = z[1]
	a = z[0]
	v0 = (b - a) / dt
	o = find_jacobian_1(model,Variable(z[0], requires_grad=True))
	k1 = torch.matmul(o,v0)
	return k1	

def make_sigma(sig):
	sigma = torch.zeros(784,20)
	for i in range(20):
		sigma[i][i] = sig[i]
	return sigma

def compute_SVD(matrix):
	u, sigma, vh = torch.svd(matrix, some=False)
	return (u, sigma, vh)

def mod(x):
	x1 = x#.numpy()
	p = 0
	for i in range(784):
		q = x1[i]
		p = p + q*q
	p = math.sqrt(p)
	return p 

def main3(z0, u0):
	x = []
	z = []
	u = []
	z.append(z0)
	u.append(u0)
	x.append(model.decode(z0))

	for i in range(0,T):
		xi = model.decode(z[len(z) - 1]).view(784)
		xi = x[len(x)-1] + 1
		ui = u[len(u) - 1].view(784)
		xiplus1 = Variable(torch.add(xi.data, dt * ui).view(784), requires_grad=True)
		zxx1, zxx2 = model.encode(xiplus1)
		# if (i > 0):
		# 	print (i)
		# 	print (xiplus1)
		# 	print (zxx1)
		ziplus1 = Variable(zxx1.data, requires_grad=True)
		xiplus1 = model.decode(ziplus1)
		Jg = find_jacobian_1(model, ziplus1)
		U, sigma, vh = compute_SVD(Jg)
		U = torch.FloatTensor(U)
		sigma = torch.FloatTensor(sigma)
		vh = torch.FloatTensor(vh)
		sigma = make_sigma(sigma)
		U, sigma, vh, jgg = reduction(U, sigma, vh, Jg)
		uiplus1 = (torch.matmul(torch.matmul(U, U.t()),u[len(u) - 1]))#.view(784,1)
		uiplus1 = (mod(u[len(u) - 1]) / mod(uiplus1)) * uiplus1
		u.append(uiplus1)
		z.append(ziplus1)
		x.append(xiplus1)
	for i in range(len(z)):
		make_image(z[i].data.view(20),"algo3_final"+(str)(i))
	make_image(z[0].data.view(20),"algo3_initial")

z0 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
z1 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
z_ = main1(model,z0,z1)
u0 = initial_velocity(z_)
main3(z0,u0)
