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


def find_u0(z0):
	sigma = np.identity(20)
	mu = np.zeros([20])
	z = z0
	z_ = z.numpy()
	a = np.matmul(np.matmul((z_ - mu).transpose(), inv(sigma)), (z_ - mu))
	#latent_space_np = (1/math.sqrt(2 * math.pi * np.linalg.det(sigma) )) * math.exp(- np.dot( (z_ - mu).transpose(), sigma, (z_ - mu) ))
	latent_space_np = np.array([a])
	#print ("hello : ",latent_space_np)	
	latent_space_ft = torch.from_numpy(latent_space_np)
	#print ("FloatTensor ",latent_space_ft)
	latent_space = Variable(latent_space_ft)
	#print ("Variable ",latent_space)
	latent_space.backward()
	# gradient = z.grad
	# print (gradient)
	# return gradient

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

zt = torch.FloatTensor(20).normal_()
(find_u0(zt))





