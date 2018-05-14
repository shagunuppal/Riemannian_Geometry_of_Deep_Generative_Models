import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

def reduction1(U, sigma, vh, x1):
	y = 0
	z = 0 
	for i in range(20):
		y += sigma[i][i]
	z = 0
	index = -1
	for i in range(20):
		z += sigma[i][i]
		x = (1.0*(z*100))/y
		if(x>=90):
			index = i
			break
	index = index + 1
	rem = 20 - index 
	if(index!=-1):
		U = U[:,:(20-rem)]
		sigma = sigma[:(20-rem),:rem]
		vh = vh[:rem,:]
		x11 = torch.mm(torch.mm(U, sigma), vh)
	if(index==-1):
		print("NO")
	return U, sigma, vh, x11


def reduction(U, sigma, vh, x1):
	y = 0
	z = 0 
	for i in range(20):
		y += sigma[i][i]
	z = 0
	index = -1
	for i in range(20):
		z += sigma[i][i]
		x = (1.0*(z*100))/y
		if(x>=90):
			index = i
			break
	index = index + 1
	rem = 20 - index 
	if(index!=-1):
		U = U[:,:(784-rem)]
		sigma = sigma[:(784-rem),:rem]
		vh = vh[:rem,:]
		x11 = torch.mm(torch.mm(U, sigma), vh)
	if(index==-1):
		print("NO")
	return U, sigma, vh, x11



