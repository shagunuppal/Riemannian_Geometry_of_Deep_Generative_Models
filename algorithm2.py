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

from algorithm1_new import *
from PCA import *

model = VAE(784,450,200,20)
load_model()

def find_v0(z):
	b = z[1]
	a = z[0]
	v0 = ((b - a)*1.0) / dt
	# v0 is 20 size float tensor 
	x1 = find_jacobian(model, Variable(z[0],requires_grad=True))
	U, sigma, vh = compute_SVD(x1)
	U = torch.FloatTensor(U)
	sigma = torch.FloatTensor(sigma)
	vh = torch.FloatTensor(vh)
	sigma = make_sigma(sigma)
	U, sigma, vh, xii = reduction1(U, sigma, vh, x1)
	v0 = torch.matmul(v0,torch.matmul(U,U.t()))
	return v0

def compute_SVD(matrix):
	u, sigma, vh = torch.svd(matrix, some=False)
	return (u, sigma, vh)

def rad2deg(rad):
	return (rad*180/math.pi)

def make_sigma(sig):
	sigma = torch.zeros(784,20)
	for i in range(20):
		sigma[i][i] = sig[i]
	return sigma

def mod(x):
	#x1 = x.numpy()
	x1 = x
	p = 0
	for i in range(784):
		q = x1[i]
		p = p + q*q
	p = math.sqrt(p)
	return p


def chhota_mod(x):
	#x1 = x.numpy()
	x1 = x
	p = 0
	for i in range(20):
		q = x1[i]
		p = p + q*q
	p = math.sqrt(p)
	return p

def find_angle(v1,v2):
	v1 = v1.view(20)
	v2 = v2.view(20)
	v = v1*v2
	v1_mod = chhota_mod(v1)
	print ("v1",v1_mod)
	v2_mod = chhota_mod(v2)
	print ("v2",v2_mod)
	num = sum(v)
	print ("sum",num)
	return (num/(v1_mod*v2_mod)) 

def main2(model,z_collection):
	u = []
	v = []
	v0 = find_v0(z_collection)
	u0 = torch.matmul(find_jacobian_1(model, Variable(z_collection[0], requires_grad=True)), v0)
	u.append(u0)
	v.append(v0)
	T  = len(z_collection) - 1
	
	for i in range (T):
		xi = model.decode(Variable(z_collection[i],requires_grad=True))
		x1 = find_jacobian_1(model, Variable(z_collection[i+1].view(20),requires_grad=True))
		U, sigma, vh = compute_SVD(x1)
		U = torch.FloatTensor(U)
		print(torch.matmul(U.t(), U))
		ui = torch.matmul(torch.matmul(U.t(), U),u[len(u) - 1].view(784,1))
		ui = (mod( u[len(u) - 1].view(784) ) / mod(ui)) * ui.view(784)
		vt_ = find_jacobian(model, Variable(z_collection[i],requires_grad=True))
		vt = torch.matmul(vt_, ui.view(784,1))
		v.append(vt)
		u.append(ui.view(784))

	ut = u[len(u) - 1]
	vt_ = find_jacobian(model, Variable(z_collection[len(z_collection) - 1],requires_grad=True))
	vt = torch.mm(vt_, ut.view(784,1))
	for i in range(len(z_collection)):
		make_image(model,z_collection[i].view(20), "algo2_latent"+(str)(i))	
	for i in range(len(v)):
		make_image(model,v[i].view(20),"algo2_tangent"+(str)(i))
		if(i!=0):
			angle = find_angle(v[i-1],v[i])
			angle = angle.data.numpy()
			print(angle)
			#angle_cos_inv = math.acos(angle)
	return vt

z0 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
z1 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)

model = load_model()

z_ = main1(model,z0,z1)
main2(model,z_collection=z_)





