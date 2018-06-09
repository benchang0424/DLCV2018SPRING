import numpy as np
import pickle
import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data as Data
from utils import *

import skvideo.io
import skimage.transform
from reader import readShortVideo
from reader import getVideoList
from sklearn.manifold import TSNE

#filepath = 'HW5_data/TrimmedVideos/video/valid'
#tag_path = 'HW5_data/TrimmedVideos/label/gt_valid.csv'
class RNN_Classifier(nn.Module):
	def __init__(self,input_size=512*7*7, hidden_size=512):
		super(RNN_Classifier, self).__init__()
		self.hidden_size = hidden_size
		self.gru = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.5)
		self.bn1 = nn.BatchNorm1d(hidden_size)
		self.fc1 = nn.Linear(hidden_size, 11)

	def forward(self, x, lengths):
		x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
		out, (hn,cn) = self.gru(x, None)
		hidden = hn[-1]
		y = self.bn1(hn[-1])
		y = F.softmax(self.fc1(y), 1)
		#y = F.relu(self.fc2(y))
		return y, hidden

class Seq_Classifier(nn.Module):
	def __init__(self,input_size=512*7*7, hidden_size=512):
		super(Seq_Classifier, self).__init__()
		self.hidden_size = hidden_size
		self.gru = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.5)
		#self.bn1 = nn.BatchNorm1d(hidden_size)
		self.fc1 = nn.Linear(hidden_size, 11)
		

	def forward(self, x, lengths):
		out_seq = []
		rnn_out, _ = self.gru(x, None)		
		for idx in range(rnn_out.size(0)):
			#rnn_fc = self.bn1(rnn_out[idx])
			category = F.softmax(self.fc1(rnn_out[idx]), 1)
			out_seq.append(category)
		
		category = torch.stack(out_seq)
		category = category.squeeze(1)
		return category

def batch_padding_rnn(train_x, train_y):
	seq_tensors = train_x.unsqueeze(1)
	label = torch.LongTensor(np.array(train_y))

	lengths = [len(train_x)]
	seq_tensors = to_var(seq_tensors)
	label = to_var(label)
	return seq_tensors, label, lengths

def batch_padding_seq(x_batch):	
	seq_tensors = x_batch.unsqueeze(1)
	lengths = [len(seq_tensors)]
	seq_tensors = to_var(seq_tensors)
	return seq_tensors, lengths	


def PlotTsneCNN(OUT_DIR):
	x_val = np.load('data/x_valid_vgg16.npy')
	y_val = np.load('data/y_valid_vgg16.npy').astype(np.long)
	print(x_val.shape)

	CNN_features_2d = TSNE(n_components=2, perplexity=40.0, random_state=24, verbose=1).fit_transform(x_val)

	cm = plt.cm.get_cmap("jet", 11)
	plt.figure(figsize=(8,6))
	plt.title('tSNE CNN features')
	plt.scatter(CNN_features_2d[:,0], CNN_features_2d[:,1], c=y_val, cmap=cm)
	plt.colorbar(ticks=range(11))
	plt.clim(-0.5, 10.5)
	filename = os.path.join(OUT_DIR, 'CNN_tsne.jpg')
	plt.savefig(filename)

	print("Plot CNN tSNE Done!")


def PlotTsneRNN(OUT_DIR):
	x_val = torch.load('data/x_valid_vgg16.pth')
	y_val = torch.load('data/y_valid_vgg16.pth')
	y_val = np.array(y_val)

	model = RNN_Classifier()
	model.load_state_dict(torch.load('RNN_model.pkt'))

	if torch.cuda.is_available():
		model.cuda()

	RNN_feautures = []
	with torch.no_grad():
		model.eval()
		for i in range(0, len(x_val)):			
			x_batch, y_batch, batch_lengths = batch_padding_rnn(x_val[i], y_val[i])
			_, hidden = model(x_batch, batch_lengths)
			RNN_feautures.append(hidden.cpu().data.numpy().reshape(-1))

		RNN_feautures = np.array(RNN_feautures)
		
	RNN_features_2d = TSNE(n_components=2, perplexity=40.0, random_state=24, verbose=1).fit_transform(RNN_feautures)

	cm = plt.cm.get_cmap("jet", 11)
	plt.figure(figsize=(8,6))
	plt.title('tSNE RNN features')
	plt.scatter(RNN_features_2d[:,0], RNN_features_2d[:,1], c=y_val, cmap=cm)
	plt.colorbar(ticks=range(11))
	plt.clim(-0.5, 10.5)
	filename = os.path.join(OUT_DIR, 'RNN_tsne.jpg')
	plt.savefig(filename)

	print("Plot RNN tSNE Done!")


def PlotSeq2Seq(video_dir, OUT_DIR):
	model = Seq_Classifier()
	model.load_state_dict(torch.load('Seq_model.pkt'))
	if torch.cuda.is_available():
		model.cuda()

	x_val = torch.load('data/x_valid_full.pth')
	y_val = torch.load('data/y_valid_full.pth')

	result = []
	acc = 0
	with torch.no_grad():
		model.eval()
		for i in range(0, len(x_val)):
			x_batch, batch_lengths = batch_padding_seq(x_val[i])
			output = model(x_batch, batch_lengths)
			predict_label = torch.argmax(output,1).cpu().data
			result.append(predict_label.numpy())

	video_name_list = sorted(os.listdir(video_dir))

	for video_num in range(len(result)):
		test = result[video_num]
		answer = y_val[video_num]
		plt.figure(figsize=(16,4))
		ax = plt.subplot(211)

		colors = plt.cm.get_cmap('Paired',11).colors
		cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in test])
		bounds = [i for i in range(len(test))]
		norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
		cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
		                                       norm=norm,
		                                       boundaries=bounds,
		                                       spacing='proportional',
		                                       orientation='horizontal')
		ax.set_ylabel('Prediction')

		ax2 = plt.subplot(212)
		cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in answer])
		bounds = [i for i in range(len(test))]
		norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
		cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
		                                       norm=norm,
		                                       boundaries=bounds,
		                                       spacing='proportional',
		                                       orientation='horizontal')


		ax2.set_ylabel('GroundTruth')

		filename = os.path.join(OUT_DIR, '{}_seq.jpg'.format(video_name_list[video_num][:-4]))
		plt.savefig(filename)
		plt.show()
	
	print("Plot seq2seq Done!")


def main(args):
	
	OUT_DIR = args.out_path
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)
	video_dir = args.video_dir


	#PlotTsneCNN(OUT_DIR)
	#PlotTsneRNN(OUT_DIR)
	PlotSeq2Seq(video_dir, OUT_DIR)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='HW5 CNN inference')
	parser.add_argument('--video_dir', help='validation video directory', type=str)
	#parser.add_argument('--csv_dir', help='ground_truth csv directory', type=str)
	parser.add_argument('--out_path', help='output figure directory', type=str)
	args = parser.parse_args()

	main(args)


#Set3