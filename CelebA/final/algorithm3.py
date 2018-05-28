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
#from torchvision.datasets import MNIST
from torch.autograd.gradcheck import zero_gradients
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import math
#from PCA import *

from algo1 import *

T = 10
dt = 1.0 / T

def initial_velocity(model,z):
	b = z[1].view(32)
	a = z[0].view(32)
	v0 = (b - a) / dt
	#v0 = Variable(v0.data.cuda())
	o = find_jacobian_1(model,Variable(z[0].data.cuda(), requires_grad=True))
	#print("o",type(o))
	#print("v0",type(v0))
	v0 = v0.data.cuda()
	#print(o.size(),v0.size())
	k1 = torch.matmul(o,v0)
	return k1	

def make_sigma(sig):
	sigma = torch.zeros(3*64*64,32).cuda()
	for i in range(32):
		sigma[i][i] = sig[i]
	return sigma

def compute_SVD(matrix):
	u, sigma, vh = torch.svd(matrix, some=False)
	return (u, sigma, vh)

def mod(x):
	x1 = x#.numpy()
	p = 0
	for i in range(3*64*64):
		q = x1[i]
		p = p + q*q
	p = math.sqrt(p)
	return p 

def main3(model,z0, u0):
	x = []
	z = []
	u = []
	z.append(z0.view(1,32))
	u.append(u0)
	z0 = z0.view(1,32)
	print("z0",z0.size())
	x.append(model.decode(z0))

	for i in range(0,T-1):
		xi = model.decode(z[len(z) - 1].view(1,32)).view(3*64*64)
		xi = x[len(x)-1]
		ui = u[len(u) - 1]
		xiplus1 = Variable(torch.add(xi.data, (dt * ui).cuda()).cuda(). view(3*64*64), requires_grad=True)
		xiplus1 = xiplus1.view(1,3,64,64)
		zxx1, zxx2 = model.encode(xiplus1)
		ziplus1 = Variable(zxx1.data.cuda(), requires_grad=True)
		xiplus1 = model.decode(ziplus1)
		Jg = find_jacobian_1(model, ziplus1)
		U, sigma, vh = compute_SVD(Jg)
		U = torch.FloatTensor(U.cpu())
		sigma = torch.FloatTensor(sigma.cpu())
		vh = torch.FloatTensor(vh.cpu())
		sigma = make_sigma(sigma)
		uiplus1 = (torch.matmul(torch.matmul(U.t(), U),u[len(u) - 1].cpu()))#.view(784,1)
		uiplus1 = (mod(u[len(u) - 1]) / mod(uiplus1)) * uiplus1
		u.append(uiplus1)
		z.append(ziplus1)
		x.append(xiplus1)
	for i in range(len(z)):
		make_image(model,Variable(z[i].data.cuda().view(1,32)),"algo3_final"+(str)(i))

model = load_model()
model.eval().cuda()
z0 = Variable(torch.FloatTensor(1,32).normal_().cuda(), requires_grad=True)
z1 = Variable(torch.FloatTensor(1,32).normal_().cuda(), requires_grad=True)
z_ = main1(model,z0,z1)
u0 = initial_velocity(model,z_)

main3(model,z0,u0)

