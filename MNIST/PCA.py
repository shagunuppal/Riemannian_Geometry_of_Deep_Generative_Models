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
		x = ((z*100.0))/y
		if(x>=95):
			index = i
			break
	index = index + 1
	rem = 20 - index 
	#print("reduction 1",index)
	if(index!=-1):
		U = U[:,:(20-rem)]
		sigma = sigma[:(20-rem),:rem]
		vh = vh[:rem,:]
		x11 = torch.mm(torch.mm(U, sigma), vh)
	if(index==-1):
		print("NO")
	return U, sigma, vh, x11


def reduction(U, sigma, vh, x1):
	sum_total = 0
	sum_current = 0 
	print("initially",U.size())
	#print(torch.matmul(U,U.t()))
	for i in range(20):
		sum_total += sigma[i][i]
	index = -1
	#print("total1",sum_total)
	for i in range(20):
		sum_current += sigma[i][i]
		x = (sum_current*100.0)/sum_total
		if(x>95):
			index = i
			break
	index = index + 1
	rem = 20 - index 
	#print("total2",sum_current)
	#print("reduction index", index)
	if(index!=-1):
		U = U[:,:(784-rem)]
		sigma = sigma[:(784-rem),:rem]
		vh = vh[:rem,:]
		x11 = torch.mm(torch.mm(U, sigma), vh)
	print("finally",U.size())
	#print(torch.matmul(U,U.t()))
	if(index==-1):
		print("NO")
	return U, sigma, vh, x11



