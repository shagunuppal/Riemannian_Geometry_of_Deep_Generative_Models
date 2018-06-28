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
import torch.onnx

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
		#print("encode",x.size())
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
		h33 = F.elu(self.fc3(z)) #20-200
		h3 = F.elu(self.fc33(h33)) #200-450
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


def load_model():
	model.load_state_dict(torch.load('./vae2.pth'))
	return model


model = load_model()
model.eval()

batch_size = 64
x = Variable(torch.FloatTensor(784).normal_(), requires_grad=True)

torch_out = torch.onnx._export(model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "model_mnsit.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file










