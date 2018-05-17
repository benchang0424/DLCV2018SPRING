import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
import pickle
import skimage.io
import skimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torchvision
from models import *
from utils import *


def plot_vae(TRAIN_DIR, TEST_DIR ,TRAIN_CSVDIR, TEST_CSVDIR, OUT_DIR):
	torch.manual_seed(424)
	latent_dim = 512
	#fig1_2
	with open("saves/vae/KLD_loss.pkl", "rb") as fp:   # Unpickling
		KLD_loss = pickle.load(fp)
	with open("saves/vae/MSE_loss.pkl", "rb") as fp:   # Unpickling
		MSE_loss = pickle.load(fp)
	plt.figure(figsize=(12, 4))
	plt.subplot(121)
	plt.title('Reconstruction (MSE) Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.plot(MSE_loss, 'r', label='mse loss')
	plt.legend(loc="best")

	plt.subplot(122)
	plt.plot(KLD_loss, 'b', label='kld loss')
	plt.title('KL Divergence Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(loc="best")
	plt.tight_layout()
	filename = os.path.join(OUT_DIR, 'fig1_2.jpg')
	plt.savefig(filename)
	
	#fig1_3
	print("Loading model...")
	model = VAE(latent_dim)
	model.load_state_dict(torch.load('saves/save_models/VAE_512.pth',map_location=lambda storage, loc: storage))

	testimg = read_image(TEST_DIR)
	test = torch.from_numpy(testimg).type(torch.FloatTensor)
	original_faces = test[:10]
	test = to_var(test)
	pred = torch.FloatTensor()
	if torch.cuda.is_available():
		model.cuda()
		pred = pred.cuda()

	print("Predicting...")
	model.eval()
	print(test.shape)
	for i in range(len(test)):
		recon, mu, logvar = model(test[i].view(1,3,64,64))
		if i < 10:
			pred = torch.cat((pred,recon.data), dim=0)

	pred = pred.cpu()
	result = torch.cat((original_faces,pred), dim=0)
	filename = os.path.join(OUT_DIR, 'fig1_3.jpg')
	torchvision.utils.save_image(result, filename, nrow=10)

	#fig1_4
	rand_variable = Variable(torch.randn(32,latent_dim),volatile=True)
	if torch.cuda.is_available():
		rand_variable = rand_variable.cuda()
	rand_output = model.decode(rand_variable)
	filename = os.path.join(OUT_DIR, 'fig1_4.jpg')
	torchvision.utils.save_image(rand_output.cpu().data, filename, nrow=8)

	#fig1_5
	# visialize the latent space
	mu, logvar = model.encode(test)
	latent_space = mu.cpu().data.numpy()
	test_attr = pd.read_csv(TEST_CSVDIR)["Male"]
	test_attr = np.array(test_attr)
	latent_emb = TSNE(n_components=2, random_state=24, verbose=1).fit_transform(latent_space)
	latent_male = latent_emb[test_attr==1]
	latent_female = latent_emb[test_attr==0]

	plt.figure()
	plt.title('Gender')
	plt.scatter(latent_female[:,0],latent_female[:,1], c='red', label='Female')
	plt.scatter(latent_male[:,0],latent_male[:,1], c='blue', label='Male')
	plt.legend(loc="best")


	filename = os.path.join(OUT_DIR, 'fig1_5.jpg')
	plt.savefig(filename)

	print("VAE Done!")


def plot_gan(TRAIN_DIR, TEST_DIR ,TRAIN_CSVDIR, TEST_CSVDIR, OUT_DIR):
	torch.manual_seed(23469)
	#fig2_2
	with open("saves/gan/D_loss.pkl", "rb") as fp:   # Unpickling
		D_loss = pickle.load(fp)
	with open("saves/gan/G_loss.pkl", "rb") as fp:   # Unpickling
		G_loss = pickle.load(fp)
	with open("saves/gan/D_real_acc.pkl", "rb") as fp:   # Unpickling
		D_real_acc = pickle.load(fp)
	with open("saves/gan/D_fake_acc.pkl", "rb") as fp:   # Unpickling
		D_fake_acc = pickle.load(fp)
	
	plt.figure(figsize=(12,4))

	plt.subplot(121)
	plt.title("Training Loss")
	plt.xlabel("epochs")
	plt.plot(D_loss, label="D loss")
	plt.plot(G_loss, label="G loss")
	plt.legend(loc="best")
	plt.subplot(122)
	plt.title("Discriminator R/F Accuracy")
	plt.xlabel("epochs")
	plt.plot(D_real_acc, label = "Real Accuracy")
	plt.plot(D_fake_acc, label = "Fake Accuracy")
	plt.legend(loc="best")

	filename = os.path.join(OUT_DIR, 'fig2_2.jpg')
	plt.savefig(filename)

	#fig2_3
	rand_inputs = Variable(torch.randn(32, 100, 1, 1),volatile=True)
	G = Generator()
	G.load_state_dict(torch.load('saves/save_models/Generator.pth',map_location=lambda storage, loc: storage))
	if torch.cuda.is_available():
		rand_inputs = rand_inputs.cuda()
		G.cuda()
	G.eval()
	rand_outputs = G(rand_inputs)
	filename = os.path.join(OUT_DIR, 'fig2_3.jpg')
	torchvision.utils.save_image(rand_outputs.cpu().data, filename, nrow=8)
	print("GAN Done!")

def plot_acgan(TRAIN_DIR, TEST_DIR ,TRAIN_CSVDIR, TEST_CSVDIR, OUT_DIR):
	torch.manual_seed(240)

	#fig3_2
	with open("saves/acgan/D_loss.pkl", "rb") as fp:   # Unpickling
		D_loss = pickle.load(fp)
	with open("saves/acgan/G_loss.pkl", "rb") as fp:   # Unpickling
		G_loss = pickle.load(fp)
	with open("saves/acgan/D_real_acc.pkl", "rb") as fp:   # Unpickling
		D_real_acc = pickle.load(fp)
	with open("saves/acgan/D_fake_acc.pkl", "rb") as fp:   # Unpickling
		D_fake_acc = pickle.load(fp)
	with open("saves/acgan/Real_attr_loss.pkl", "rb") as fp:   # Unpickling
		Real_attr_loss = pickle.load(fp)
	with open("saves/acgan/Fake_attr_loss.pkl", "rb") as fp:   # Unpickling
		Fake_attr_loss = pickle.load(fp)
	
	plt.figure(figsize=(12,4))
	plt.subplot(121)
	plt.title("Training loss of Attribute Classification")
	plt.xlabel("epochs")
	plt.plot(Real_attr_loss, label="Real classification loss")
	plt.plot(Fake_attr_loss, label="Fake classification loss")
	plt.legend(loc="best")

	plt.subplot(122)
	plt.title("Discriminator R/F Accuracy")
	plt.xlabel("epochs")
	plt.plot(D_real_acc, label = "Real Accuracy")
	plt.plot(D_fake_acc, label = "Fake Accuracy")
	plt.legend(loc="best")

	filename = os.path.join(OUT_DIR, 'fig3_2.jpg')
	plt.savefig(filename)

#fig3_3
	fixed_noise = torch.randn((10, 100, 1, 1))
	fixed_noise = torch.cat((fixed_noise,fixed_noise), dim=0)
	fixed_label = torch.cat((torch.zeros((10,1,1,1)), torch.ones((10,1,1,1))), dim=0)
	fixed_noise = torch.cat((fixed_noise,fixed_label), dim=1)
	fixed_noise = to_var(fixed_noise)
	G = ACGenerator()
	G.load_state_dict(torch.load('saves/save_models/ACGenerator.pth',map_location=lambda storage, loc: storage))
	if torch.cuda.is_available():
		G.cuda()
	G.eval()
	rand_outputs = G(fixed_noise)
	filename = os.path.join(OUT_DIR, 'fig3_3.jpg')
	torchvision.utils.save_image(rand_outputs.cpu().data, filename, nrow=10)
	print("ACGAN Done!")


def main(args):
	TRAIN_DIR = os.path.join(args.train_path, 'train')
	TEST_DIR = os.path.join(args.train_path, 'test')
	TRAIN_CSVDIR = os.path.join(args.train_path, 'train.csv')
	TEST_CSVDIR = os.path.join(args.train_path, 'test.csv')
	OUT_DIR = args.out_path
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)
	if args.mode=='vae':
		plot_vae(TRAIN_DIR, TEST_DIR ,TRAIN_CSVDIR, TEST_CSVDIR, OUT_DIR)
	if args.mode=='gan':
		plot_gan(TRAIN_DIR, TEST_DIR ,TRAIN_CSVDIR, TEST_CSVDIR, OUT_DIR)
	if args.mode=='acgan':
		plot_acgan(TRAIN_DIR, TEST_DIR ,TRAIN_CSVDIR, TEST_CSVDIR, OUT_DIR)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='HW4 plot figure')
	parser.add_argument('--train_path', help='training data directory', type=str)
	parser.add_argument('--out_path', help='output figure directory', type=str)
	parser.add_argument('--mode', help='output figure directory', type=str)
	args = parser.parse_args()
	main(args)