import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data as Data
from utils import *
from models import *
import random


BATCH_SIZE = 1

def batch_padding(x_batch, y_batch, mode='train'):
	# x_batch's shape : [n,25088]
	# y_batch's shape : [n]
	if mode == 'train':
		if len(x_batch) > 1500:
			max_sample = 512
		else:
			max_sample = 350
		selected_idx = sorted(random.sample([i for i in range(0, x_batch.size(0))], max_sample))
		
		seq_tensors = [x_batch[i] for i in selected_idx]
		label = [y_batch[i] for i in selected_idx]

		#print(seq_tensors.shape)
		#print(label.shape)
		seq_tensors = torch.stack(seq_tensors)
		label = torch.stack(label)
		#print("1 : ", seq_tensors.shape)
		#print("2 : ", label.shape)
		seq_tensors = seq_tensors.unsqueeze(1)
		lengths = [len(seq_tensors)]
		seq_tensors = to_var(seq_tensors)
		label = to_var(label)

	elif mode == 'valid':
		#seq_tensors = torch.stack(x_batch)
		#label = torch.stack(y_batch)
		label = torch.LongTensor(np.array(y_batch))
		seq_tensors = x_batch.unsqueeze(1)
		lengths = [len(seq_tensors)]
		seq_tensors = to_var(seq_tensors)
		label = to_var(label)
	return seq_tensors, label, lengths



def train(n_epochs, x_train, y_train, x_val, y_val):
	model = Seq_Classifier()
	model.load_state_dict(torch.load('save_models/RNN_weight222/RNN_0.4874274661508704.pkt'))
	optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999))
	loss_function = nn.CrossEntropyLoss()

	if torch.cuda.is_available():
		model.cuda()

	Train_loss_list, Train_acc_list = [],[]
	Val_loss_list, Val_acc_list = [],[]
	best_accuracy = 0.0


	for epoch in range(n_epochs):
		start = time.time()
		train_loss = 0.0
		train_acc = 0.0
		val_loss = 0.0
		val_acc = 0.0
		train_cnt = 0
		val_cnt = 0
		idx = np.random.permutation(len(x_train))
		x_train_shuffle = [x_train[i] for i in idx]
		y_train_shuffle = [y_train[i] for i in idx] 	
			
		for idx in range(0, len(x_train_shuffle)):
			train_cnt += 1
			x_input = x_train_shuffle[idx]
			y_input = y_train_shuffle[idx]

			x_batch, y_batch, batch_lengths = batch_padding(x_input, y_input, mode='train')

			model.train()
			optimizer.zero_grad()
			output = model(x_batch, batch_lengths)

			batch_loss = loss_function(output, y_batch)
			batch_loss.backward()
			optimizer.step()
			
			train_loss += batch_loss.item()
			output_label = torch.argmax(output,1).cpu()
			Acc = np.mean((output_label == y_batch.cpu()).numpy())
			train_acc += Acc

			if idx > 0:		
				print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| CE_Loss: {:.6f} | Acc: {:.6f} | Time: {}  '.format(
					epoch+1, idx, len(x_train),
					100. * idx / len(x_train),
					batch_loss, Acc,
					timeSince(start, idx/ len(x_train))),end='')
		
		
		#Validation
		with torch.no_grad():
			for batch_idx in range(0, len(x_val)):
				val_cnt += 1
				x_input = x_val[batch_idx]
				y_input = y_val[batch_idx]			
				x_batch, y_batch, batch_lengths = batch_padding(x_input, y_input, mode='valid')
				
				model.eval()
				output = model(x_batch, batch_lengths)
				batch_loss = loss_function(output, y_batch)
				predict_label = torch.argmax(output,1).cpu()
				acc = np.mean((predict_label == y_batch.cpu()).numpy())
				val_loss += batch_loss.item()
				val_acc += acc


		print('\nTrain:\nCE_Loss: {:.6f} | Accuracy: {:.6f} \nValidation:\nCE_loss: {:.6f} | Accuracy: {:.6f}'.format(
			train_loss/train_cnt, train_acc/len(x_train), val_loss/val_cnt, val_acc/len(x_val)))

		Train_loss_list.append(train_loss/train_cnt)
		Train_acc_list.append(train_acc/len(x_train))
		Val_loss_list.append(val_loss/val_cnt)
		Val_acc_list.append(val_acc/len(x_val))
		
		#Checkpoint
		if (val_acc > best_accuracy):
			best_accuracy = val_acc
			torch.save(model.state_dict(), 'save_models/Seq_b1/Seq_'+str(val_acc/len(x_val))+'.pkt')
			print ('Saving Improved Model(val_acc = %.6f)...' % (val_acc/len(x_val)))

		print('-'*88)

	with open('./checkpoint/Q3/train_loss.pkl', 'wb') as fp:
		pickle.dump(Train_loss_list, fp)
	with open('./checkpoint/Q3/train_acc.pkl', 'wb') as fp:
		pickle.dump(Train_acc_list, fp)
	with open('./checkpoint/Q3/val_loss.pkl', 'wb') as fp:
		pickle.dump(Val_loss_list, fp)
	with open('./checkpoint/Q3/val_acc.pkl', 'wb') as fp:
		pickle.dump(Val_acc_list, fp)

def main():
	x_train = torch.load('data/x_train_full.pth')
	y_train = torch.load('data/y_train_full.pth')
	x_val = torch.load('data/x_valid_full.pth')
	y_val = torch.load('data/y_valid_full.pth')

	#print(x_train[0].shape)		torch.Size([2994, 25088])
	#print(y_train[0].shape)		torch.Size([2994])
	#for i in range(len(x_train)):
	#	print(x_train[i].shape)
	print("load data done !!!")

	train(100, x_train, y_train, x_val, y_val)

	print("training done !!!")

	

if __name__ == '__main__':
	main()