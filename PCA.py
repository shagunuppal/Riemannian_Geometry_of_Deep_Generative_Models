import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

def PCA(U, sigma, vh, x1):
	y = 1000
	j = 0
	x2 = x1
	for i in range(1,20):
		U = U[:784,:784-i]
		sigma = sigma[:784-i,:20-i]
		vh = vh[:20-i,:]
		x11 = torch.mm(torch.mm(U, sigma), vh)
		new1 = (abs(x1 - x11)) / ((abs)(x1))
		new_  = torch.mean(new1)*100
		#print(new_)
		#print("yoyo_new:",j,new_.data)
		j+=1
		if(new_<y):
			y = new_
			x2 = x11
	#print(x2.size())
	return x2, U, sigma, vh

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
		#print(x)
		if(x>=90):
			index = i
			#print(index)
			break
	if(index!=0):
		U = U[:784,:index]
		sigma = sigma[:index,:index]
		vh = vh[:index,:]
		x11 = torch.mm(torch.mm(U, sigma), vh)
	if(index==-1):
		print("NO")
	return U, sigma, vh, x11



