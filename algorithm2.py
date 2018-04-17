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
from numpy.linalg import inv

from algorithm1 import *

model = VAE(784,400,20)
load_model()


def find_v0(z0):
	z = Variable(z0.view(20,1),requires_grad=True)
	sigma = torch.FloatTensor(np.identity(20))
	mu = torch.zeros(20,1)
	sigma_np = sigma.numpy()
	det = np.linalg.det(sigma_np)
	c = 1 / (math.sqrt(2 * math.pi * det)) 
	a = torch.exp(torch.mm(torch.t(z),z))
	latent_space = (a * c)
	latent_space.backward()
	k = z.grad.data
	return k	

def compute_SVD(matrix):
	u, sigma, vh = np.linalg.svd(matrix, full_matrices = True)
	return (u, sigma, vh)

def main2(z_collection):
	u = []
	v0 = find_v0(z_collection[0])
	#print(v0)
	#print("aaaaaaa",z_collection[0])
	u0 = torch.matmul(find_jacobian_1(model, Variable(z_collection[0], requires_grad=True)), v0)
	u.append(u0)
	T  = len(z_collection) - 1
	
	for i in range (T):
		xi = model.decode(Variable(z_collection[i],requires_grad=True))
		x1 = find_jacobian_1(model, Variable(z_collection[i+1],requires_grad=True))
		U, sigma, vh = compute_SVD(x1)
		U = torch.FloatTensor(U)
		ui = torch.mm(torch.mm(U.t(), U),u[len(u) - 1].view(784,1))
		ui = (find_mod(u[len(u) - 1]) / find_mod(ui)) * ui
		u.append(ui)

	ut = u[len(u) - 1]
	vt_ = find_jacobian(model, Variable(z_collection[len(z_collection) - 1],requires_grad=True))
	vt = torch.mm(vt_, ut)
	make_image(vt.view(20),"algo2_final")
	make_image(v0.view(20),"algo2_initial")
	return vt

zt = torch.FloatTensor(20).normal_().view(20,1)
z0 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
z1 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)

z_ = main1(model,z0,z1)
main2(z_collection=z_)





