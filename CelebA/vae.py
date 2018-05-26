import os, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
#import imageio
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

# parser = argparse.ArgumentParser(description='VAE CelebA Example')
# parser.add_argument('--batch-size', type=int, default=100, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=1, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')

#args = parser.parse_args()

num_epochs = 1
batch_size = 100
learning_rate = 0.0002

mean = Variable(torch.zeros(100,32))
log_variance = Variable(torch.zeros(100,32))

mg_transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*4*4,256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc21 = nn.Linear(256,32) # for mean
        self.fc22 = nn.Linear(256,32) # for standard deviation
        
        ###################################################################
        
        self.fc3 = nn.Linear(32, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 64*4*4)
        self.bn7 = nn.BatchNorm1d(64*4*4)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        # additional 
        # self.bn11 = nn.BatchNorm2d(3)
        # self.deconv5 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=0)

    def encode(self, x):
        h1 = F.elu(self.bn1(self.conv1(x)))
        #print("h1:",h1.size())
        h2 = F.elu(self.bn2(self.conv2(h1)))
        #print("h2:",h2.size())
        h3 = F.elu(self.bn3(self.conv3(h2)))
        #print("h3:",h3.size())
        h4 = F.elu(self.bn4(self.conv4(h3)))
        #print("h4:",h4.size())
        h4 = h4.view(-1, self.num_flat_features(h4))
        #print("h4",h4.size())
        h5 = F.elu(self.fc1(h4))
        #print("h5",h5.size())
        return (self.fc21(h5)), F.sigmoid(self.fc22(h5))

    def decode(self, z):
        #print("z",z.size())
        h6 = F.elu(self.bn6(self.fc3(z)))
        print("h6",h6.size())
        h7 = F.elu(self.bn7(self.fc4(h6)))
        print("h7",h7.size())
        #ll = h7.size()[0]
        #print(ll)
        h7 = h7.view(h7.size()[0],64,4,4)
        h8 = F.elu(self.bn8(self.deconv1(h7))) 
        print ("h8",h8.size())
        h9 = F.elu(self.bn9(self.deconv2(h8)))
        print ("h9",h9.size())
        h10 = F.elu(self.bn10(self.deconv3(h9)))
        print("h10",h10.size())
        h11 = (self.deconv4(h10))
        print ("h11",h11.size())
        return h11

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def forward(self, x):
        global mean
        global log_variance
        mu, logvar = self.encode(x)
        mean = mu
        log_variance = logvar
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features

model = VAE()
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
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_set):
            img, _ = data
            #img = img.view(img.size(0), -1)
            img = Variable(img)
            optimizer.zero_grad()
            #print("image here",img.size()) # 64*64*3
            recon_batch, mu, logvar = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            #running_loss.append(loss.data[0])
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
            save = to_img(recon_batch.data)
            save_image(save, './vae_img/image_{}.png'.format(epoch))
        #log_value('recon_loss', np.average(running_loss),epoch)
    return model

def load_model():
    model.load_state_dict(torch.load('./vae.pth'))
    return model

def save_model(model):
    torch.save(model.state_dict(), './vae.pth')

def generate_image():
    z = torch.FloatTensor(100,32).normal_()
    make_image(z,"generated-image")

def make_image(model,z,name):
    x = model.decode(Variable(z.data, requires_grad = True))
    x = x.view(3,64,64)
    img = x.data.numpy()
    #i = randint(0,100)
    #img = img[:,:,:]
    x1 = img[0,:,:]
    x2 = img[1,:,:]
    x3 = img[2,:,:]
    #print(x1)
    img_final = np.zeros([64,64,3])
    img_final[:,:,0] = x1
    img_final[:,:,1] = x2
    img_final[:,:,2] = x3
    #print("img2",img_final[:,:,0])
    plt.imshow(img_final, interpolation = 'nearest')
    plt.savefig('./' + name + '.jpg')


#############################################################################
# TRAINING A NEW MODEL
train(batchsize = batch_size)
save_model(model)
#############################################################################

#############################################################################
# LOADING EXISTING MODEL
#load_model()
#############################################################################

#generate_image()

