# Algorithm 1 : Geodesic Path

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

import vae_cuda

import torch._utils
try:
	torch._utils._rebuild_tensor_v2
except AttributeError:
	def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
		tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
		tensor.requires_grad = requires_grad
		tensor._backward_hooks = backward_hooks
		return tensor
	torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


model = load_model()
T = 10
dt = 1.0 / T
epsilon = 2500
z_collection = []
delta_e = torch.FloatTensor(32,64*64*3).zero_()

def find_mod1(x):
    # x is float tensor
    p = 0
    x = x.data
    x1 = x.numpy()
    for i in range(32):
        q = x1[i]
        p += q*q
    return math.sqrt(p)

def linear_distance(z1,z2):
    x = z2 - z1
    return find_mod1(x)

def linear_interpolation(model,z0, zt):
	z_collection.append(z0)
	for i in range(T-2):
		z0n = z_collection[len(z_collection)-1] + (zt-z0)*dt
		z_collection.append(z0n)
		print("distance_"+(str)(i+1),linear_distance(z_collection[len(z_collection)-2],z_collection[len(z_collection)-1])) 
		print("arclength_"+(str)(i+1),arc_length(model, z_collection[len(z_collection)-2],z_collection[len(z_collection)-1]))   
	z_collection.append(zt) 
	print("distance_"+(str)(T-1),linear_distance(z_collection[len(z_collection)-2],z_collection[len(z_collection)-1]))  
	print("arc_length"+(str)(T-1),arc_length(model, z_collection[len(z_collection)-2],z_collection[len(z_collection)-1])) 

def find_jacobian(model, z1): #Jh
	z = z1
	dec = Variable(model.decode(z).data, requires_grad=True)
	enc1, enc2 = model.encode(dec)
	jacobian = torch.FloatTensor(20,784).zero_()
	for j in range(20):
		f = torch.FloatTensor(20).zero_()
		f[j] = 1	
		enc1.backward(f, retain_graph=True)
		jacobian[j,:] = dec.grad.data
		dec.grad.data.zero_()
	return jacobian

def find_jacobian_1(model, z1): #Jg
	z = z1
	dec = model.decode(z)
	jacobian = torch.FloatTensor(784,20).zero_()
	for j in range(784):
		f = torch.FloatTensor(784).zero_()
		f[j] = 1	
		dec.backward(f, retain_graph=True)
		jacobian[j,:] = z.grad.data
		z.grad.data.zero_()
	return jacobian



def find_energy(model,z0, z1, z2):
	a11 = find_jacobian_1(model,Variable(z1, requires_grad=True))
	a1 = torch.transpose(find_jacobian_1(model,Variable(z1, requires_grad=True)),0,1)
	a2 = ((model.decode(Variable(z2)) - 2*model.decode(Variable(z1))+model.decode(Variable(z0))).data).view(784,1)
	e = -(1 / dt)*(torch.mm(a1,a2))
	return e

def find_etta_i(model,z0,z1,z2):
	dt = 1/T
	z0 = z0.view(20)
	z1 = z1.view(20)
	z2 = z2.view(20)
	a1 = find_jacobian(model,Variable(z1))
	x1 = model.decode(Variable(z2))
	x2 = 2*model.decode(Variable(z1))
	x3 = model.decode(Variable(z0))
	a21 = (x1-x2+x3).data
	a2 = a21.view(784,1)
	e = -(1 / dt)*torch.mm(a1,a2)
	return e

def find_mod(x):
	# x is float tensor
	p = 0
	x = x.data
	x1 = x.numpy()
	for i in range(20):
		q = x1[i]
		p += q*q
	return p

def find_mod1(x):
	# x is float tensor
	p = 0
	x = x.data
	x1 = x.numpy()
	for i in range(784):
		q = x1[i]
		p += q*q
	return math.sqrt(p)

def sum_energy(model):
	delta_e = torch.FloatTensor(20,784).zero_()
	for i in range(1,T-2):
		delta_e += find_etta_i(model,z_collection[i-1],z_collection[i],z_collection[i+1])
	multi = (torch.mm((delta_e),torch.transpose(delta_e,0,1)))
	return multi

def sum_energy_1(model):
	delta_e = torch.FloatTensor(20,1).zero_()
	for i in range(1,T-2):
		delta_e += find_energy(model,z_collection[i-1].view(20),z_collection[i].view(20),z_collection[i+1].view(20))
	return find_mod(delta_e)

def make_image(model,z,name):
	x = model.decode(Variable(z))
	#print("decoded",x)
	x = x.view(28,28)
	img = x.data.numpy()
	plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
	plt.savefig('./' + name + '.jpg')

def arc_length(model, z1, z2):
	xx = 0 
	xx = model.decode(z2) - model.decode(z1)
	xx1 = find_mod1(xx)
	return xx1 * T

def geodesic_length(model, z_collection):
	xx = 0
	for i in range(1,T):
		xx1 = model.decode(z_collection[i]) -  model.decode(z_collection[i-1]) 
		xx += find_mod1(xx1)*T
	return xx

def main1(model,z0,zt):
	step_size = 0.1
	y = linear_distance(z0,zt)
	print("distance_ends:",y)
	linear_interpolation(model,z0,zt)
	print("geodesic_ends:",geodesic_length(model, z_collection))
	# while (sum_energy_1(model) > epsilon):
	# 	print(sum_energy_1(model))
	# 	for i in range(1,T-1):
	# 		etta_i = find_etta_i(model, z_collection[i-1], z_collection[i], z_collection[i+1])
	# 		e1 = step_size*etta_i
	# 		z_collection[i] = z_collection[i].view(20,1)
	# 		z_collection[i] = z_collection[i] - e1
	# for p in range(T):
	# 	make_image(model,z=z_collection[p].view(20),name=str(p))
	# return z_collection


z0 = Variable(torch.FloatTensor(32).normal_().cuda(), requires_grad=True)
zt = Variable(torch.FloatTensor(32).normal_().cuda(), requires_grad=True)

main1(model=model,z0=z0, zt=zt)


	

		















