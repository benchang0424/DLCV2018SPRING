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
	fixed_noise = torch.randn((10,latent_dim, 1, 1))
	fixed_noise = torch.cat((fixed_noise,fixed_noise),0)
	fixed_label = torch.cat((torch.zeros((10,1,1,1)), torch.ones((10,1,1,1))),0)
	fixed_noise = torch.cat((fixed_noise,fixed_label),1)
	fixed_noise = to_var(fixed_noise)

	G = ACGenerator()
	D = ACDiscriminator()
	if torch.cuda.is_available():
		G.cuda()
		D.cuda()

	# setup optimizer
	optimizerD = optim.Adam(D.parameters(), lr=0.0003, betas=(0.5, 0.999))
	optimizerG = optim.Adam(G.parameters(), lr=0.0003, betas=(0.5, 0.999))
	
	criterion_dis = nn.BCELoss()	# smiling or not
	criterion_aux = nn.BCELoss()



	D_loss_list = []
	G_loss_list = []
	D_real_acc_list = []
	D_fake_acc_list = []
	Real_attr_loss_list = []
	Fake_attr_loss_list = []
	print("START training...")

	for epoch in range(n_epochs):
		start = time.time()
		D_total_loss = 0.0
		G_total_loss = 0.0
		Real_aux_total_loss = 0.0
		Fake_aux_total_loss = 0.0
		Real_total_acc = 0.0
		Fake_total_acc = 0.0

		for batch_idx, (data, real_class) in enumerate(train_loader):
			batch_size = len(data)
			#real_class = torch.FloatTensor(real_class).view(-1,1,1,1)
			#real_class = real_class.view(-1,1,1,1)
			data = to_var(data)
			real_class = to_var(real_class)
			real_labels = to_var(torch.ones(batch_size))
			fake_labels = to_var(torch.zeros(batch_size))
			
			# ================================================================== #
			#                      Train the discriminator                       #
			# ================================================================== #
			
			# Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
			# Second term of the loss is always zero since real_labels == 1
			D.zero_grad()
			output_dis, output_aux = D(data)
			D_loss_real_dis = criterion_dis(output_dis, real_labels)
			D_loss_real_aux = criterion_aux(output_aux, real_class)
			D_accu_real = np.mean((output_dis > 0.5).cpu().data.numpy())
			D_loss_real = D_loss_real_dis + D_loss_real_aux
			D_loss_real.backward()

			# Compute BCELoss using fake images
			# First term of the loss is always zero since fake_labels == 0
			noise = torch.randn(batch_size, latent_dim, 1, 1)
			fake_class = torch.from_numpy(np.random.randint(2, size=batch_size)).type(torch.FloatTensor).view(batch_size,1,1,1)
			z = torch.cat((noise,fake_class),1)
			z = to_var(z)
			fake_class = to_var(fake_class)
			fake_images = G(z)
			output_dis, output_aux = D(fake_images.detach())
			D_loss_fake_dis = criterion_dis(output_dis, fake_labels)
			D_loss_fake_aux = criterion_aux(output_aux, fake_class)
			D_accu_fake = np.mean((output_dis < 0.5).cpu().data.numpy())
			D_loss_fake = D_loss_fake_dis + D_loss_fake_aux
			D_loss_fake.backward()

			D_loss = D_loss_real_dis + D_loss_fake_dis
			D_total_loss += D_loss.data[0]
			Real_aux_total_loss += D_loss_real_aux.data[0]
			Fake_aux_total_loss += D_loss_fake_aux.data[0]
			Real_total_acc += D_accu_real
			Fake_total_acc += D_accu_fake

			#D_loss_back = D_loss_real + D_loss_fake
			#D_loss_back.backward()
			optimizerD.step()

			# ================================================================== #
			#                        Train the generator                         #
			# ================================================================== #
			
			# Compute loss with fake images
			# We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
			G.zero_grad()
			noise = torch.randn(batch_size, latent_dim, 1, 1)
			fake_class = torch.from_numpy(np.random.randint(2, size=batch_size)).type(torch.FloatTensor).view(batch_size,1,1,1)
			z = torch.cat((noise,fake_class),1)
			z = to_var(z)
			fake_class = to_var(fake_class)
			fake_images = G(z)
			output_dis, output_aux = D(fake_images)

			G_loss_fake_dis = criterion_dis(output_dis, real_labels)
			G_loss_fake_aux = criterion_aux(output_aux, fake_class)
			G_loss = G_loss_fake_dis + G_loss_fake_aux
			G_total_loss += G_loss.data[0]
			G_loss.backward()
			optimizerG.step()

			if batch_idx % 5 == 0:
				
				print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| D_Loss: {:.6f} , G_loss: {:.6f}| Real_Acc: {:.6f} , Fake_Acc: {:.6f}| Time: {}  '.format(
					epoch+1, (batch_idx+1) * len(data), len(train_loader.dataset),
					100. * batch_idx * len(data)/ len(train_loader.dataset),
					D_loss.data[0] / len(data), G_loss.data[0] / len(data),
					D_accu_real, D_accu_fake,
					timeSince(start, (batch_idx+1)*len(data)/ len(train_loader.dataset))), end='')
				
				#sys.stdout.write('\033[K')
	
		print('\n====> Epoch: {} \nD_loss: {:.6f} |  G_loss: {:.6f} \nReal_Aux_Loss: {:.6f} |  Fake_Aux_Loss: {:.6f} \nReal_Acc: {:.6f} |  Fake_Acc: {:.6f}'.format(
			epoch+1, D_total_loss/len(train_loader), G_total_loss/len(train_loader),
			Real_aux_total_loss/len(train_loader), Fake_aux_total_loss/len(train_loader),
			Real_total_acc/len(train_loader), Fake_total_acc/len(train_loader)))
		print('-'*88)

		D_loss_list.append(D_total_loss/len(train_loader))
		G_loss_list.append(G_total_loss/len(train_loader))
		D_real_acc_list.append(Real_total_acc/len(train_loader))
		D_fake_acc_list.append(Fake_total_acc/len(train_loader))
		Real_attr_loss_list.append(Real_aux_total_loss/len(train_loader))
		Fake_attr_loss_list.append(Fake_aux_total_loss/len(train_loader))

		G.eval()
		rand_outputs = G(fixed_noise)
		G.train()
		torchvision.utils.save_image(rand_outputs.cpu().data,
								'./output_imgs/acgan/fig3_3_%03d.jpg' %(epoch+1), nrow=10)
		
	torch.save(G.state_dict(), './saves/save_models/ACGenerator.pth')
	torch.save(D.state_dict(), './saves/save_models/ACDiscriminator.pth')

	with open('./saves/acgan/D_loss.pkl', 'wb') as fp:
		pickle.dump(D_loss_list, fp)
	with open('./saves/acgan/G_loss.pkl', 'wb') as fp:
		pickle.dump(G_loss_list, fp)
	with open('./saves/acgan/D_real_acc.pkl', 'wb') as fp:
		pickle.dump(D_real_acc_list, fp)
	with open('./saves/acgan/D_fake_acc.pkl', 'wb') as fp:
		pickle.dump(D_fake_acc_list, fp)
	with open('./saves/acgan/Real_attr_loss.pkl', 'wb') as fp:
		pickle.dump(Real_attr_loss_list, fp)
	with open('./saves/acgan/Fake_attr_loss.pkl', 'wb') as fp:
		pickle.dump(Fake_attr_loss_list, fp)


def main(args):
	#TRAIN_DIR = "./hw4_data/train/"
	#TEST_DIR = "./hw4_data/test/"
	#TRAIN_CSVDIR = "./hw4_data/train.csv"
	#TEST_CSVDIR = "./hw4_data/test.csv"
	TRAIN_DIR = os.path.join(args.train_path, 'train')
	TEST_DIR = os.path.join(args.train_path, 'test')
	TRAIN_CSVDIR = os.path.join(args.train_path, 'train.csv')
	TEST_CSVDIR = os.path.join(args.train_path, 'test.csv')

	train_data = CelebADataset('ACGAN',TRAIN_DIR, TEST_DIR, TRAIN_CSVDIR, TEST_CSVDIR)

	print("Read Data Done !!!")
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
	print("Enter Train")
	train(200, train_loader)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='VAE Example')
	parser.add_argument('--train_path', help='training data directory', type=str)
	args = parser.parse_args()
	main(args)

