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
import skimage.io
import skimage
from models import *
from utils import *
import pickle
import sys
torch.manual_seed(424)
BATCH_SIZE = 128
latent_dim = 100


def train(n_epochs, train_loader):
	
	rand_inputs = Variable(torch.randn(32,latent_dim, 1, 1),volatile=True)

	G = Generator()
	D = Discriminator()
	if torch.cuda.is_available():
		rand_inputs = rand_inputs.cuda()
		G.cuda()
		D.cuda()

	# setup optimizer
	optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
	criterion = nn.BCELoss()


	D_loss_list = []
	G_loss_list = []
	D_real_acc_list = []
	D_fake_acc_list = []

	print("START training...")

	for epoch in range(n_epochs):
		start = time.time()
		D_total_loss = 0.0
		G_total_loss = 0.0
		Real_total_acc = 0.0
		Fake_total_acc = 0.0
		for batch_idx, (data, _) in enumerate(train_loader):
			batch_size = len(data)
			real_labels = torch.ones(batch_size)
			fake_labels = torch.zeros(batch_size)
			data = to_var(data)
			real_labels = to_var(real_labels)
			fake_labels = to_var(fake_labels)
			
			# ================================================================== #
			#                      Train the discriminator                       #
			# ================================================================== #
			
			# Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
			# Second term of the loss is always zero since real_labels == 1
			D.zero_grad()
			outputs = D(data)
			D_loss_real = criterion(outputs, real_labels)
			D_accu_real = np.mean((outputs > 0.5).cpu().data.numpy())

			# Compute BCELoss using fake images
			# First term of the loss is always zero since fake_labels == 0
			z = torch.randn(batch_size, latent_dim, 1, 1)
			z = to_var(z)
			fake_images = G(z)
			outputs = D(fake_images.detach())
			D_loss_fake = criterion(outputs, fake_labels)
			D_accu_fake = np.mean((outputs < 0.5).cpu().data.numpy())

			# Backprop and optimize
			D_loss = D_loss_real + D_loss_fake
			D_total_loss += D_loss.data[0]
			D_loss.backward()
			optimizerD.step()

			# ================================================================== #
			#                        Train the generator                         #
			# ================================================================== #
			
			# Compute loss with fake images
			# We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
			G.zero_grad()
			z = torch.randn(batch_size, latent_dim, 1, 1)
			z = to_var(z)
			fake_images = G(z)
			outputs = D(fake_images)
			G_loss = criterion(outputs, real_labels)
			G_total_loss += G_loss.data[0]
			G_loss.backward()
			optimizerG.step()

			Real_total_acc += D_accu_real
			Fake_total_acc += D_accu_fake
			if batch_idx % 5 == 0:		
				print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| D_Loss: {:.6f} , G_loss: {:.6f}| Real_Acc: {:.6f} , Fake_Acc: {:.6f}| Time: {}  '.format(
					epoch+1, (batch_idx+1) * len(data), len(train_loader.dataset),
					100. * batch_idx * len(data)/ len(train_loader.dataset),
					D_loss.data[0] / len(data), G_loss.data[0] / len(data),
					D_accu_real, D_accu_fake,
					timeSince(start, (batch_idx+1)*len(data)/ len(train_loader.dataset))),end='')
	
		print('\n====> Epoch: {} \nD_loss: {:.6f} | Real_Acc: {:.6f} \nG_loss: {:.6f} | Fake_Acc: {:.6f}'.format(
			epoch+1, D_total_loss/len(train_loader.dataset), Real_total_acc/len(train_loader),
			G_total_loss/len(train_loader.dataset), Fake_total_acc/len(train_loader)))
		print('-'*88)

		D_loss_list.append(D_total_loss/len(train_loader.dataset))
		G_loss_list.append(G_total_loss/len(train_loader.dataset))
		D_real_acc_list.append(Real_total_acc/len(train_loader))
		D_fake_acc_list.append(Fake_total_acc/len(train_loader))

		G.eval()
		rand_outputs = G(rand_inputs)
		G.train()
		torchvision.utils.save_image(rand_outputs.cpu().data,
								'./output_imgs/gan/fig2_3_%03d.jpg' %(epoch+1), nrow=8)
		
		torch.save(G.state_dict(), './saves/save_models/Generator_%03d.pth'%(epoch+1))
		torch.save(D.state_dict(), './saves/save_models/Discriminator_%03d.pth'%(epoch+1))

	with open('./saves/gan/D_loss.pkl', 'wb') as fp:
		pickle.dump(D_loss_list, fp)
	with open('./saves/gan/G_loss.pkl', 'wb') as fp:
		pickle.dump(G_loss_list, fp)
	with open('./saves/gan/D_real_acc.pkl', 'wb') as fp:
		pickle.dump(D_real_acc_list, fp)
	with open('./saves/gan/D_fake_acc.pkl', 'wb') as fp:
		pickle.dump(D_fake_acc_list, fp)


def main(args):	
	#TRAIN_DIR = "./hw4_data/train/"
	#TEST_DIR = "./hw4_data/test/"
	#TRAIN_CSVDIR = "./hw4_data/train.csv"
	#TEST_CSVDIR = "./hw4_data/test.csv"
	TRAIN_DIR = os.path.join(args.train_path, 'train')
	TEST_DIR = os.path.join(args.train_path, 'test')
	TRAIN_CSVDIR = os.path.join(args.train_path, 'train.csv')
	TEST_CSVDIR = os.path.join(args.train_path, 'test.csv')

	train_data = CelebADataset('GAN',TRAIN_DIR, TEST_DIR, TRAIN_CSVDIR, TEST_CSVDIR)

	print("Read Data Done !!!")
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
	print("Enter Train")
	train(150, train_loader)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='GAN Example')
	parser.add_argument('--train_path', help='training data directory', type=str)
	args = parser.parse_args()
	main(args)

