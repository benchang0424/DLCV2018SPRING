import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNN_Classifier(nn.Module):
	def __init__(self):
		super(CNN_Classifier, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(512*7*7, 4096),
			nn.ReLU(),
			nn.Linear(4096, 1024),
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Linear(256,11),
			nn.Softmax(1)
		)

	def forward(self, x):
		output = self.main(x)
		return output


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
		y = self.bn1(hn[-1])
		y = F.softmax(self.fc1(y), 1)
		#y = F.relu(self.fc2(y))
		return y


class RNN_Classifier222(nn.Module):
	def __init__(self,input_size=512*7*7, hidden_size=512):
		super(RNN_Classifier222, self).__init__()
		self.hidden_size = hidden_size
		self.gru = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.5)
		#self.bn1 = nn.BatchNorm1d(hidden_size)
		self.fc1 = nn.Linear(hidden_size, 11)

	def forward(self, x, lengths):
		x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
		out, (hn,cn) = self.gru(x, None)
		#y = self.bn1(hn[-1])
		y = F.softmax(self.fc1(hn[-1]), 1)
		#y = F.relu(self.fc2(y))
		return y


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
		

if __name__ == '__main__':
	model = RNN_Classifier()
	print(model)
