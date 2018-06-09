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
from models import *
import skvideo.io
import skimage.transform
from reader import readShortVideo
from reader import getVideoList

#filepath = 'HW5_data/TrimmedVideos/video/valid'
#tag_path = 'HW5_data/TrimmedVideos/label/gt_valid.csv'

def Get_data(video_path, tag_path):
	model = torchvision.models.vgg16(pretrained=True).features
	if torch.cuda.is_available():
		model.cuda()
	file_dict = getVideoList(tag_path)
	x, y = [], []
	print(len(file_dict['Video_index']))
	with torch.no_grad():
		for i in range(len(file_dict['Video_index'])):
			frames = readShortVideo(video_path, file_dict['Video_category'][i],file_dict['Video_name'][i])		
			if frames.shape[0] > 120:
				output_1 = model(torch.from_numpy(frames[0:120,:,:,:]).cuda()).detach().cpu().reshape(-1,512*7*7)
				output_2 = model(torch.from_numpy(frames[120:,:,:,:]).cuda()).detach().cpu().reshape(-1,512*7*7)
				output = torch.cat((output_1, output_2), 0)
			else:
				output = model(torch.from_numpy(frames).cuda()).detach().cpu().reshape(-1,512*7*7)

			output = torch.mean(output, 0).numpy()
			x.append(output)
			y.append(int(file_dict['Action_labels'][i]))
			print('\rreading image from {}...{}'.format(video_path,i),end='')

	print('\rreading image from {}...finished'.format(video_path))
	
	return np.array(x).astype(np.float32), np.array(y).astype(np.uint8)

def PlotLearningCurve(OUT_DIR):
	with open('./checkpoint/Q1/train_loss.pkl', 'rb') as fp:
		train_loss = pickle.load(fp)
	with open('./checkpoint/Q1/train_acc.pkl', 'rb') as fp:
		train_acc = pickle.load(fp)
	with open('./checkpoint/Q1/val_acc.pkl', 'rb') as fp:
		val_acc = pickle.load(fp)
	with open('./checkpoint/Q1/val_loss.pkl', 'rb') as fp:
		val_loss = pickle.load(fp)
		
	plt.figure(figsize=(12, 4))
	plt.subplot(121)
	plt.title('Training/Validation Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.plot(train_loss, label='Train Loss')
	plt.plot(val_loss, label='Valid Loss')
	plt.legend(loc="best")

	plt.subplot(122)
	plt.title('Training/Validation Accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.plot(train_acc, label='Train Accuracy')
	plt.plot(val_acc, label='Valid Accuracy')
	plt.legend(loc="best")

	filename = os.path.join(OUT_DIR, 'fig1_2.jpg')
	plt.savefig(filename)

	print("Done!")


def Predict(x_val, y_val, OUT_DIR):
	model = torch.load('CNN_model.pkl')
	x_val = to_var(x_val)
	y_val = to_var(y_val)
	if torch.cuda.is_available():
		model.cuda()

	result, acc = [],[]
	with torch.no_grad():
		model.eval()
		output = model(x_val)
		predict_label = torch.argmax(output,1).cpu().data
		acc.append((predict_label == y_val.cpu()).numpy())
		result.append(predict_label.numpy())

	print(len(acc))
	acc = np.mean(acc)
	print("validation acc : ", acc)
	result = result[0]
	print(result.shape)
	
	filename = os.path.join(OUT_DIR, 'p1_valid.txt')
	cnt=0
	with open(filename, 'w') as f:
		for item in result:
			f.write("%s\n" % item)
			cnt += 1
			if cnt != len(result):
				f.write("%s\n" % item)
			else:
				f.write("%s" % item)

	print("Predict Done!")


def main(args):
	OUT_DIR = args.out_path
	video_dir = args.video_dir
	csv_dir = args.csv_dir
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	x_val, y_val = Get_data(video_dir, csv_dir)
	x_val = torch.from_numpy(x_val)
	x_val = x_val.view(x_val.size(0), -1)
	y_val = y_val.astype(np.long)
	y_val = torch.from_numpy(y_val).view(-1)

	print("get data done !!!")

	#PlotLearningCurve(OUT_DIR)
	Predict(x_val, y_val, OUT_DIR)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='HW5 CNN inference')
	parser.add_argument('--video_dir', help='validation video directory', type=str)
	parser.add_argument('--csv_dir', help='ground_truth csv directory', type=str)
	parser.add_argument('--out_path', help='output figure directory', type=str)
	args = parser.parse_args()

	main(args)