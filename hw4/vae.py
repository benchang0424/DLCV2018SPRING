import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import optim
import torchvision
import time
import argparse
import pickle
import skimage.io
import skimage
import os
from models import *
from utils import *
import torchvision

torch.manual_seed(424)
BATCH_SIZE = 128
latent_dim = 512

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    MSE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, KLD, MSE

def train(n_epochs, train_loader, test_loader):

    rand_variable = Variable(torch.randn(32, latent_dim), volatile=True)

    model = VAE(latent_dim)
    if torch.cuda.is_available():
        rand_variable = rand_variable.cuda()
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    MSE_loss_list = []
    KLD_loss_list = []
    print("START training...")
    model.train()
    
    for epoch in range(n_epochs):
        start = time.time()
        MSE_loss = 0.0
        KLD_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = to_var(data)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, KLD, MSE = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            MSE_loss += MSE.data[0]
            KLD_loss += KLD.data[0]
            optimizer.step()
            if batch_idx % 5 == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f} | Time: {}  '.format(
                    epoch+1, (batch_idx+1) * len(data), len(train_loader.dataset),
                    100. * batch_idx * len(data)/ len(train_loader.dataset),
                    loss.data[0] / len(data),
                    timeSince(start, (batch_idx+1) * len(data) / len(train_loader.dataset))), end='')
            
        print('\n====> Epoch: {} \nMSE_loss: {:.6f}, \nKLD_loss: {:.6f}'.format(
            epoch+1, MSE_loss/(12288*len(train_loader.dataset)),KLD_loss/len(train_loader)))

        MSE_loss_list.append(MSE_loss/(12288*len(train_loader.dataset)))
        KLD_loss_list.append(KLD_loss/len(train_loader))

        model.eval()
        test_mseloss = 0.0
        for i, (data, _) in enumerate(test_loader):
            data = Variable(data, volatile=True)
            if torch.cuda.is_available():
                data = data.cuda()
            recon_batch, mu, logvar = model(data)
            _, _, MSE = loss_function(recon_batch, data, mu, logvar)
            test_mseloss += MSE.data[0]
            if i == 0:
                comparison = torch.cat([data, recon_batch.view(-1, 3, 64, 64)])
                torchvision.utils.save_image(comparison.cpu().data, './output_imgs/vae_512/fig1_3_%02d.jpg' %(epoch+1),nrow=10)

        print('====> Test set MSE loss: {:.4f}'.format(test_mseloss/(12288*len(test_loader.dataset))))
        print('-'*80)
        rand_output = model.decode(rand_variable)
        torchvision.utils.save_image(rand_output.cpu().data, './output_imgs/vae_512/fig1_4_%03d.jpg' %(epoch+1),nrow=8)
        model.train()
    torch.save(model.state_dict(), './saves/save_models/VAE_512.pth')
    with open('./saves/vae/MSE_loss_512.pkl', 'wb') as fp:
        pickle.dump(MSE_loss_list, fp)
    with open('./saves/vae/KLD_loss_512.pkl', 'wb') as fp:
        pickle.dump(KLD_loss_list, fp)
    #the_model = TheModelClass(*args, **kwargs)
    #the_model.load_state_dict(torch.load(PATH))


def main(args):
    #TRAIN_DIR = "./hw4_data/train/"
    #TEST_DIR = "./hw4_data/test/"
    #TRAIN_CSVDIR = "./hw4_data/train.csv"
    #TEST_CSVDIR = "./hw4_data/test.csv"
    TRAIN_DIR = os.path.join(args.train_path, 'train')
    TEST_DIR = os.path.join(args.train_path, 'test')
    TRAIN_CSVDIR = os.path.join(args.train_path, 'train.csv')
    TEST_CSVDIR = os.path.join(args.train_path, 'test.csv')

    train_data = CelebADataset('VAE_train', TRAIN_DIR, TEST_DIR, TRAIN_CSVDIR, TEST_CSVDIR)
    test_data = CelebADataset('VAE_test', TRAIN_DIR, TEST_DIR, TRAIN_CSVDIR, TEST_CSVDIR)
    print("Read Data Done !!!")

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=1)
    print("Enter Train")
    
    train(200, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Example')
    parser.add_argument('--train_path', help='training data directory', type=str)
    args = parser.parse_args()

    main(args)
