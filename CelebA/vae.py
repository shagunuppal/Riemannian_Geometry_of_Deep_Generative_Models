import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import transforms

################################################################################################################################################################################

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
    x = x.view(x.size(0), 1, 64, 64)
    return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*2*2,256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc21 = nn.Linear(256,32) # for mean
        self.fc22 = nn.Linear(256,32) # for standard deviation
        
        ###################################################################
        
        self.fc3 = nn.Linear(32, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 64*2*2)
        self.bn7 = nn.BatchNorm1d(64*2*2)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=0)
        self.bn8 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0)
        self.bn9 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=0)
        self.bn10 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=0)
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
        h6 = F.elu(self.bn6(self.fc3(z)))
        h7 = F.elu(self.bn7(self.fc4(h6)))
        h7 = h7.view(100,64,2,2)
        h8 = F.elu(self.bn8(self.deconv1(h7))) 
        #print ("h8",h8.size())
        h9 = F.elu(self.bn9(self.deconv2(h8)))
        #print ("h9",h9.size())
        h10 = F.elu(self.bn10(self.deconv3(h9)))
        #print("h10",h10.size())
        h11 = (self.deconv4(h10))
        #print ("h11",h11.size())
        #h12 = self.deconv5(h11)
        #print ("h12",h12.size())
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

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(batchsize):
    #train_set = torch.utils.data.DataLoader(datasets.MNIST('./data',train=True,download=True,transform=transforms.ToTensor()),batch_size=batchsize, shuffle=True)
    data_dir = 'data/resized_celebA/' # this path depends on your computer
    dset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    train_set = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_set):
            img, _ = data
            #img = img.view(img.size(0), -1)
            img = Variable(img)
            optimizer.zero_grad()
            print(img.size()) # 64*64*3
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
    model.load_state_dict(torch.load('./vae.pth'))
    return model

def save_model(model):
    torch.save(model.state_dict(), './vae.pth')


#############################################################################
# TRAINING A NEW MODEL
train(batchsize = batch_size)
save_model(model)
#############################################################################

#############################################################################
# LOADING EXISTING MODEL
#load_model()
#############################################################################




################################################################################################################################################################################

# G(z)
# class generator(nn.Module):
#     # initializers
#     def __init__(self, d=128):
#         super(generator, self).__init__()
#         self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
#         self.deconv1_bn = nn.BatchNorm2d(d*8)
#         self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
#         self.deconv2_bn = nn.BatchNorm2d(d*4)
#         self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
#         self.deconv3_bn = nn.BatchNorm2d(d*2)
#         self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
#         self.deconv4_bn = nn.BatchNorm2d(d)
#         self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

#     # forward method
#     def forward(self, input):
#         # x = F.relu(self.deconv1(input))
#         x = F.relu(self.deconv1_bn(self.deconv1(input)))
#         x = F.relu(self.deconv2_bn(self.deconv2(x)))
#         x = F.relu(self.deconv3_bn(self.deconv3(x)))
#         x = F.relu(self.deconv4_bn(self.deconv4(x)))
#         x = F.tanh(self.deconv5(x))

#         return x

# class discriminator(nn.Module):
#     # initializers
#     def __init__(self, d=128):
#         super(discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
#         self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
#         self.conv2_bn = nn.BatchNorm2d(d*2)
#         self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
#         self.conv3_bn = nn.BatchNorm2d(d*4)
#         self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
#         self.conv4_bn = nn.BatchNorm2d(d*8)
#         self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

#     # forward method
#     def forward(self, input):
#         x = F.leaky_relu(self.conv1(input), 0.2)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
#         x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
#         x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
#         x = F.sigmoid(self.conv5(x))

#         return x

# def normal_init(m, mean, std):
#     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(mean, std)
#         m.bias.data.zero_()

# fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
# #fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)

# def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
#     z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
#     #z_ = Variable(z_.cuda(), volatile=True)

#     G.eval()
#     if isFix:
#         test_images = G(fixed_z_)
#     else:
#         test_images = G(z_)
#     G.train()

#     size_figure_grid = 5
#     fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
#     for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)

#     for k in range(5*5):
#         i = k // 5
#         j = k % 5
#         ax[i, j].cla()
#         ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

#     label = 'Epoch {0}'.format(num_epoch)
#     fig.text(0.5, 0.04, label, ha='center')
#     plt.savefig(path)

#     if show:
#         plt.show()
#     else:
#         plt.close()

# def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
#     x = range(len(hist['D_losses']))

#     y1 = hist['D_losses']
#     y2 = hist['G_losses']

#     plt.plot(x, y1, label='D_loss')
#     plt.plot(x, y2, label='G_loss')

#     plt.xlabel('Iter')
#     plt.ylabel('Loss')

#     plt.legend(loc=4)
#     plt.grid(True)
#     plt.tight_layout()

#     if save:
#         plt.savefig(path)

#     if show:
#         plt.show()
#     else:
#         plt.close()

# # training parameters
# batch_size = 128
# lr = 0.0002
# train_epoch = 20

# # data_loader
# img_size = 64
# isCrop = False
# if isCrop:
#     transform = transforms.Compose([
#         transforms.Scale(108),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# else:
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])


# data_dir = 'data/resized_celebA/' # this path depends on your computer
# dset = datasets.ImageFolder(data_dir, transform)


# train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)
# temp = plt.imread(train_loader.dataset.imgs[0][0])
# if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
#     #here()
#     sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
#     sys.exit(1)

# # network
# G = generator(128)
# D = discriminator(128)
# G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)
# # G.cuda()
# # D.cuda()

# # Binary Cross Entropy loss
# BCE_loss = nn.BCELoss()

# # Adam optimizer
# G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# # results save folder
# if not os.path.isdir('CelebA_DCGAN_results'):
#     os.mkdir('CelebA_DCGAN_results')
# if not os.path.isdir('CelebA_DCGAN_results/Random_results'):
#     os.mkdir('CelebA_DCGAN_results/Random_results')
# if not os.path.isdir('CelebA_DCGAN_results/Fixed_results'):
#     os.mkdir('CelebA_DCGAN_results/Fixed_results')

# train_hist = {}
# train_hist['D_losses'] = []
# train_hist['G_losses'] = []
# train_hist['per_epoch_ptimes'] = []
# train_hist['total_ptime'] = []

# print('Training start!')
# start_time = time.time()
# for epoch in range(train_epoch):
#     D_losses = []
#     G_losses = []

#     # learning rate decay
#     if (epoch+1) == 11:
#         G_optimizer.param_groups[0]['lr'] /= 10
#         D_optimizer.param_groups[0]['lr'] /= 10
#         print("learning rate change!")

#     if (epoch+1) == 16:
#         G_optimizer.param_groups[0]['lr'] /= 10
#         D_optimizer.param_groups[0]['lr'] /= 10
#         print("learning rate change!")

#     num_iter = 0

#     epoch_start_time = time.time()
#     for x_, _ in train_loader:
#         # train discriminator D
#         D.zero_grad()
        
#         if isCrop:
#             x_ = x_[:, :, 22:86, 22:86]

#         mini_batch = x_.size()[0]

#         y_real_ = torch.ones(mini_batch)
#         y_fake_ = torch.zeros(mini_batch)

#         x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
#         D_result = D(x_).squeeze()
#         D_real_loss = BCE_loss(D_result, y_real_)

#         z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
#         z_ = Variable(z_.cuda())
#         G_result = G(z_)

#         D_result = D(G_result).squeeze()
#         D_fake_loss = BCE_loss(D_result, y_fake_)
#         D_fake_score = D_result.data.mean()

#         D_train_loss = D_real_loss + D_fake_loss

#         D_train_loss.backward()
#         D_optimizer.step()

#         D_losses.append(D_train_loss.data[0])

#         # train generator G
#         G.zero_grad()

#         z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
#         z_ = Variable(z_.cuda())

#         G_result = G(z_)
#         D_result = D(G_result).squeeze()
#         G_train_loss = BCE_loss(D_result, y_real_)
#         G_train_loss.backward()
#         G_optimizer.step()

#         G_losses.append(G_train_loss.data[0])

#         num_iter += 1

#     epoch_end_time = time.time()
#     per_epoch_ptime = epoch_end_time - epoch_start_time


#     print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
#                                                               torch.mean(torch.FloatTensor(G_losses))))
#     p = 'CelebA_DCGAN_results/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#     fixed_p = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#     show_result((epoch+1), save=True, path=p, isFix=False)
#     show_result((epoch+1), save=True, path=fixed_p, isFix=True)
#     train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
#     train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
#     train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
