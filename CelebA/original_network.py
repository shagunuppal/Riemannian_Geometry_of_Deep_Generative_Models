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

num_epochs = 1
batch_size = 100
learning_rate = 0.0002

mean = Variable(torch.zeros(100,32).cuda())
log_variance = Variable(torch.zeros(100,32).cuda())

mg_transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('./vae_img_1'):
    os.mkdir('./vae_img_1')

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 32, args.intermediate_size)

        # Latent space
        self.fc21 = nn.Linear(args.intermediate_size, args.hidden_size)
        self.fc22 = nn.Linear(args.intermediate_size, args.hidden_size)

        # Decoder
        self.fc3 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.fc4 = nn.Linear(args.intermediate_size, 8192)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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
            #img = img.view(img.size(0), -1)
            img = Variable(img.cuda())
            optimizer.zero_grad()
            print("image here",img.size()) # 64*64*3
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
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img_1/image_{}.png'.format(epoch))
        #log_value('recon_loss', np.average(running_loss),epoch)
    return model

def load_model():
    model.load_state_dict(torch.load('./vae_1.pth'))
    return model

def save_model(model):
    torch.save(model.state_dict(), './vae_1.pth')

def generate_image():
    z = torch.FloatTensor(100,32).normal_()
    make_image(z,"generated-image")

def make_image(model,z,name):
    x = model.decode(Variable(z.data.cuda(), requires_grad = True))
    x = x.view(3,64,64)
    img = x.cpu().data.numpy()
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

