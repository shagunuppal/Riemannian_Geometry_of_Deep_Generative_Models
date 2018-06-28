#from tensorboard_logger import configure, log_value
import os, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
from torchvision import transforms
from torchvision.utils import save_image
from random import randint
import numpy as np

#from tensorboard_logger import configure, log_value
#configure('logs/' + 'CelebA_loss')
#log_value('recon_loss', 1.0, 0)

################################################################################################################################################################################

parser = argparse.ArgumentParser(description='VAE CelebA Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                     help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                     help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                     help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                     help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                     help='how many batches to wait before logging training status')

args = parser.parse_args()

num_epochs = 30 
batch_size = 100
learning_rate = 0.0002

#configure('./logs/' + 'elu_conv')
#log_value('recon_loss',1.0,0)

img_transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.nc = nc
        # self.ngf = ngf
        # self.ndf = ndf
        # self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.e2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.e3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.e4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)

        # self.e5 = nn.Conv2d(64, ndf*8, 4, 2, 1)
        # self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(64*4*4, 32)
        self.fc2 = nn.Linear(64*4*4, 32)

        # decoder
        self.d1 = nn.Linear(32, 64*4*4)

        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(64)

        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn7 = nn.BatchNorm2d(32)

        # self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.bn8 = nn.BatchNorm2d(32)

        # self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        #self.bn9 = nn.BatchNorm2d(3, 1.e-3)

        # self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd5 = nn.ReplicationPad2d(1)
        # self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ELU()
	    ####
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        #print("h1",h1.size())
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        #print("h2",h2.size())
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        #print("h3",h3.size())
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        #print("h4",h4.size())
	    #h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h4.view(-1, 64*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        #i args.cuda:
        #    eps = torch.cuda.FloatTensor(std.size()).normal_()
        #else:
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h11 = self.relu(self.d1(z))
        ####
        #print(h11.size())
        h11 = h11.view(h11.size()[0], 64, 4, 4)
        #print(h11.size())
        h22 = self.leakyrelu(self.bn6(self.d2((h11))))
        #print(h22.size())
        h33 = self.leakyrelu(self.bn7(self.d3((h22))))
        #print(h33.size())
        h44 = self.leakyrelu(self.bn8(self.d4((h33))))
        #h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
        #print(h44.size())
        return self.sigmoid(self.d5(h44))

    #def get_latent_var(self, x):
        #mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        #z = self.reparametrize(mu, logvar)
        #return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar



model = VAE().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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


def train(batchsize):
    data_dir = 'data/resized_celebA/' # this path depends on your computer
    dset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    train_set = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        #running_loss = []
        model.train().cuda()
        train_loss = 0
        for batch_idx, data in enumerate(train_set):
            img, _ = data
            img = Variable(img.cuda())
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
	    #running_loss.append(loss.data[0])
            ########################################
            #array.append([epoch, loss.data[0] / len(img), 100. * batch_idx / len(train_set)])
                # epoch, loss, percentage
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_set.dataset)))
        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img/image_{}.png'.format(epoch))
        #log_value('recon_loss', np.average(running_loss),epoch)
    return model

def load_model():
    model.load_state_dict(torch.load('./vae_new.pth'))
    return model

def save_model(model):
    torch.save(model.state_dict(), './vae_new.pth')

def generate_image(model):
    z = Variable(torch.FloatTensor(1,32).normal_(), requires_grad=True)
    print("z",z)
    z1 = Variable(torch.FloatTensor(1,32).normal_(), requires_grad=True)
    print("z1",z1)
    make_image(model, z,"generated-image")
    make_image(model, z1, "generated-image1")

def make_image(model,z,name):
    x = model.decode(Variable(z.data.cuda(), requires_grad = True))
    x = x.view(1,3,64,64)
    img = x.data.cpu().numpy()
    x1 = img[0,0,:,:]
    x2 = img[0,1,:,:]
    x3 = img[0,2,:,:]
    img_final = np.zeros([64,64,3])
    img_final[:,:,0] = x1
    img_final[:,:,1] = x2
    img_final[:,:,2] = x3
    plt.imshow(img_final, interpolation = 'nearest')
    plt.savefig('./' + name + '.jpg')


#############################################################################
# TRAINING A NEW MODEL
#train(batchsize = batch_size)
#save_model(model)
#############################################################################

#############################################################################
# LOADING EXISTING MODEL
#model = load_model()
#############################################################################
#model.eval().cuda()
#generate_image(model)
#z0 = Variable(torch.FloatTensor(1,32).normal_().cuda(), requires_grad=True)
#zt = Variable(torch.FloatTensor(1,32).normal_().cuda(), requires_grad=True)
#model.decode(z0)
#model.decode(zt)
#model.encode(z0)
#model.encode(zt)

