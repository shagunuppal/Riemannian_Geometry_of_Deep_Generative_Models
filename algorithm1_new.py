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

from distance import *

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


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

mean = Variable(torch.zeros(128,20))
log_variance = Variable(torch.zeros(128,20))

img_transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('./vae_img'):
	os.mkdir('./vae_img')

def to_img(x):
	x = x.clamp(0, 1)
	x = x.view(x.size(0), 1, 28, 28)
	return x

class VAE(nn.Module):
	def __init__(self,n1,n2,n3,latent_dimension):
		super(VAE, self).__init__()

		self.fc1 = nn.Linear(n1, n2)
		self.fc11 = nn.Linear(n2,n3)
		self.fc21 = nn.Linear(n3, latent_dimension)
		self.fc22 = nn.Linear(n3, latent_dimension)
		self.fc3 = nn.Linear(latent_dimension, n3)
		self.fc33 = nn.Linear(n3,n2)
		self.fc4 = nn.Linear(n2, n1)

	def encode(self, x):
		# h
		h11 = F.relu(self.fc1(x))
		h1 = F.relu(self.fc11(h11))
		return self.fc21(h1), self.fc22(h1)

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def decode(self, z):
		# g
		h33 = F.relu(self.fc3(z)) #20-200
		h3 = F.relu(self.fc33(h33)) #200-450
		return F.sigmoid(self.fc4(h3)) #450-784

	def get_latent_variable(self, mu, logvar):
		z = self.reparametrize(mu, logvar)
		return z

	def forward(self, x):
		global mean
		global log_variance
		mu, logvar = self.encode(x)
		mean = mu
		log_variance = logvar
		z = self.reparametrize(mu, logvar)
		return self.decode(z), mu, logvar

model = VAE(784,450,200,20)

reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
	"""
	recon_x: generating images
	x: origin images
	mu: latent mean
	logvar: latent log variance
	"""
	BCE = reconstruction_function(recon_x, x)  # mse loss
	# loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element).mul_(-0.5)
	# KL divergence
	return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(batchsize):
	train_set = torch.utils.data.DataLoader(datasets.MNIST('./data',train=True,download=True,transform=transforms.ToTensor()),batch_size=batchsize, shuffle=True)

	for epoch in range(num_epochs):
		model.train()
		train_loss = 0
		for batch_idx, data in enumerate(train_set):
			img, _ = data
			img = img.view(img.size(0), -1)
			img = Variable(img)
			optimizer.zero_grad()
			recon_batch, mu, logvar = model(img)
			loss = loss_function(recon_batch, img, mu, logvar)
			loss.backward()
			train_loss += loss.data[0]
			optimizer.step()
			if batch_idx % 100 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch,
					batch_idx * len(img),
					len(train_set.dataset), 
					100. * batch_idx / len(train_set),
					loss.data[0] / len(img)))

			########################################
			#array.append([epoch, loss.data[0] / len(img), 100. * batch_idx / len(train_set)])
				# epoch, loss, percentage
		print('====> Epoch: {} Average loss: {:.4f}'.format(
			epoch, train_loss / len(train_set.dataset)))
		if epoch % 10 == 0:
			save = to_img(recon_batch.cpu().data)
			save_image(save, './vae_img/image_{}.png'.format(epoch))
	return model

def load_model():
	model.load_state_dict(torch.load('./vae2.pth'))
	return model

def save_model(model):
	torch.save(model.state_dict(), './vae2.pth')

def linear_interpolation(model,z0, zt):
	z_collection.append(z0)
	print(z0)
	print("hii",model.decode(z_collection[0]))
	for i in range(T-2):
		z0n = z_collection[len(z_collection)-1] + (zt-z0)*dt
		z_collection.append(z0n)
		#print("distance_"+(str)(i+1),linear_distance(z_collection[len(z_collection)-2],z_collection[len(z_collection)-1])) 
		print("arclength_"+(str)(i+1),arc_length(model, z_collection[len(z_collection)-2],z_collection[len(z_collection)-1]))   
	z_collection.append(zt) 
	#print("distance_"+(str)(T-1),linear_distance(z_collection[len(z_collection)-2],z_collection[len(z_collection)-1]))  
	#print("arc_length"+(str)(T-1),arc_length(model, z_collection[len(z_collection)-2],z_collection[len(z_collection)-1])) 

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

# def find_jacobian_1(model, z1):
# 	z = z1
# 	dec = model.decode(z)
# 	jacobian = torch.FloatTensor(784,20).zero_()
# 	for j in range(784):
# 		n = dec[j]
# 		n.backward(retain_graph=True)
# 		jacobian[j,:] = z.grad.data
# 	print ("J",jacobian)
# 	return jacobian


T = 4
dt = 1.0 / T
epsilon = 1000
z_collection = []
delta_e = torch.FloatTensor(20,784).zero_()

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
	x = x.view(28,28)
	img = x.data.numpy()
	plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
	plt.savefig('./' + name + '.jpg')

def make_image_1(x,name):
	img = x.numpy().reshape(28,28)
	plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
	plt.savefig('./' + name + '.jpg')

def arc_length(model, z1, z2):
	xx = 0 
	print("there",model.decode(z_collection[0]))
	xx = model.decode(z_collection[1]) - model.decode(z_collection[0])
	#print("final",model.decode(z_collection[1]))
	xx1 = find_mod1(xx)
	return xx1 * T

def geodesic_length(model, z_collection):
	xx = 0
	print("first",model.decode(z_collection[1]))
	for i in range(1,T):
		xx = model.decode(z_collection[1]) -  model.decode(z_collection[0]) 
		xx = find_mod1(xx)*T
		#print("first",model.decode(z_collection[1])) 
	return xx

def main1(model,z0,zt):
	step_size = 0.1
	print("aa gaya")
	y = linear_distance(z0,zt)
	#print("distance_ends:",y)
	print("yahan")
	linear_interpolation(model,z0,zt)
	print("here",model.decode(z_collection[0]))
	#print("arclength:",arc_length(model,z_collection[1],z_collection[0]))
	#print("geodesic_ends:",geodesic_length(model, z_collection))
	while (sum_energy_1(model) > epsilon):
		print(sum_energy_1(model))
		for i in range(1,T-1):
			etta_i = find_etta_i(model, z_collection[i-1], z_collection[i], z_collection[i+1])
			e1 = step_size*etta_i
			z_collection[i] = z_collection[i].view(20,1)
			z_collection[i] = z_collection[i] - e1
	# for p in range(T):
	# 	make_image(model,z=z_collection[p].view(20),name=str(p))
	return z_collection

#############################################################################
# TRAINING A NEW MODEL
# train(batchsize = batch_size)
# save_model(model)
#############################################################################

#############################################################################
# LOADING EXISTING MODEL
load_model()
#############################################################################

z0 = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)
zt = Variable(torch.FloatTensor(20).normal_(), requires_grad=True)

#main1(model=model,z0=z0, zt=zt)


	

		















