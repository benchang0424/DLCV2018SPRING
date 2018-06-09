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

batch_size = 64


def train(n_epochs, train_loader, x_val, y_val):
	model = CNN_Classifier()
	optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.5, 0.999))
	loss_function = nn.CrossEntropyLoss()
	x_val = to_var(x_val)
	y_val = to_var(y_val)
	if torch.cuda.is_available():
		model.cuda()

	train_loss_list, train_acc_list = [],[]
	val_loss_list, val_acc_list = [],[]
	best_accuracy = 0.0
	for epoch in range(n_epochs):
		start = time.time()
		CE_loss = 0.0
		Train_Acc = 0.0
		for batch_idx, (x,y) in enumerate(train_loader):
			batch_size = x.size(0)
			x = to_var(x)
			y = to_var(y)

			model.train()
			optimizer.zero_grad()

			output = model(x)
			err = loss_function(output, y)
			#print (output.size())	#[64,11]
			#print (y.size())		#[64]
			err.backward()
			optimizer.step()
			
			CE_loss += err.item()
			output_label = torch.argmax(output,1).cpu()
			Acc = np.mean((output_label == y.cpu()).numpy())
			Train_Acc += Acc



			if batch_idx % 2 == 0:		
				print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| CE_Loss: {:.6f} | Acc: {:.6f} | Time: {}  '.format(
					epoch+1, (batch_idx+1) * len(x), len(train_loader.dataset),
					100. * batch_idx * len(x)/ len(train_loader.dataset),
					CE_loss, Acc,
					timeSince(start, (batch_idx+1)*len(x)/ len(train_loader.dataset))),end='')
		
		
		#Validation
		with torch.no_grad():
			model.eval()
			output = model(x_val)
			val_loss = loss_function(output, y_val)
			predict_label = torch.argmax(output,1).cpu()
			val_acc = np.mean((predict_label == y_val.cpu()).numpy())

		print('\n====> Epoch: {} \nTrain:\nCE_Loss: {:.6f} | Accuracy: {:.6f} \nValidation:\nCE_loss: {:.6f} | Accuracy: {:.6f}'.format(
			epoch+1, CE_loss/len(train_loader), Train_Acc/len(train_loader), val_loss, val_acc))

		train_loss_list.append(CE_loss/len(train_loader))
		train_acc_list.append(Train_Acc/len(train_loader))
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)
		#Checkpoint
        	#torch.save(model.state_dict(), "./models/CNN_FC_model.pkt")
		if (val_acc > best_accuracy):
			best_accuracy = val_acc
			torch.save(model, 'save_models/CNN/CNN_'+str(epoch)+'.pkl')
			print ('Saving Improved Model(val_acc = %.6f)...' % (val_acc))

		print('-'*88)

	with open('./checkpoint/Q1/train_loss.pkl', 'wb') as fp:
		pickle.dump(train_loss_list, fp)
	with open('./checkpoint/Q1/train_acc.pkl', 'wb') as fp:
		pickle.dump(train_acc_list, fp)
	with open('./checkpoint/Q1/val_loss.pkl', 'wb') as fp:
		pickle.dump(val_loss_list, fp)
	with open('./checkpoint/Q1/val_acc.pkl', 'wb') as fp:
		pickle.dump(val_acc_list, fp)

def main():
	x_train = np.load('data/x_train_vgg16.npy')
	y_train = np.load('data/y_train_vgg16.npy').astype(np.long)
	x_val = np.load('data/x_valid_vgg16.npy')
	y_val = np.load('data/y_valid_vgg16.npy').astype(np.long)

	x_train = torch.from_numpy(x_train)
	y_train = torch.from_numpy(y_train).view(-1)
	x_val = torch.from_numpy(x_val)
	y_val = torch.from_numpy(y_val).view(-1)
	x_train = x_train.view(x_train.size(0), -1)
	x_val = x_val.view(x_val.size(0), -1)


	print(x_train.size)
	print(y_train.size)
	print(x_val.size)
	print(y_val.size)

	dataset = Data.TensorDataset(x_train, y_train)
	train_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
	print("load data done!!!")
	train(100, train_loader, x_val, y_val)

	

if __name__ == '__main__':
	main()