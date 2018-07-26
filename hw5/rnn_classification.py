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


BATCH_SIZE = 32

def batch_padding(train_x_batch, train_y_batch):
    
    seq_lengths = torch.LongTensor(list(map(len, train_x_batch)))
    
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensors = [train_x_batch[i] for i in perm_idx]
    seq_tensors = nn.utils.rnn.pad_sequence(seq_tensors)
    label = torch.LongTensor(np.array(train_y_batch)[perm_idx])
    seq_tensors = to_var(seq_tensors)
    label = to_var(label)
    
    return seq_tensors, label, seq_lengths


def train(n_epochs, x_train, y_train, x_val, y_val):
    model = RNN_Classifier()
    optimizer = torch.optim.Adam(model.parameters(), slr=0.0001, betas=(0.9, 0.999))
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
            
        for batch_idx in range(0, len(x_train_shuffle), BATCH_SIZE):
            train_cnt += 1
            if batch_idx+BATCH_SIZE > len(x_train_shuffle):
                x_input = x_train_shuffle[batch_idx:]
                y_input = y_train_shuffle[batch_idx:]
            else:
                x_input = x_train_shuffle[batch_idx:batch_idx+BATCH_SIZE]
                y_input = y_train_shuffle[batch_idx:batch_idx+BATCH_SIZE]

            x_batch, y_batch, batch_lengths = batch_padding(x_input,y_input)
            
            model.train()
            optimizer.zero_grad()

            output = model(x_batch, batch_lengths)
            batch_loss = loss_function(output, y_batch)
            #print (output.size())  #[64,11]
            #print (y.size())       #[64]
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
            output_label = torch.argmax(output,1).cpu()
            Acc = np.sum((output_label == y_batch.cpu()).numpy())
            train_acc += Acc

            if batch_idx > 0:       
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| CE_Loss: {:.6f} | Acc: {:.6f} | Time: {}  '.format(
                    epoch+1, batch_idx, len(x_train),
                    100. * batch_idx / len(x_train),
                    batch_loss, Acc/len(x_input),
                    timeSince(start, batch_idx/ len(x_train))),end='')

        #Validation
        with torch.no_grad():
            for batch_idx in range(0, len(x_val), BATCH_SIZE):
                val_cnt += 1
                if batch_idx+BATCH_SIZE > len(x_val):
                    x_input = x_val[batch_idx:]
                    y_input = y_val[batch_idx:]
                else:
                    x_input = x_val[batch_idx:batch_idx+BATCH_SIZE]
                    y_input = y_val[batch_idx:batch_idx+BATCH_SIZE]

                x_batch, y_batch, batch_lengths = batch_padding(x_input,y_input)
                model.eval()
                output = model(x_batch, batch_lengths)
                batch_loss = loss_function(output, y_batch)
                predict_label = torch.argmax(output,1).cpu()
                acc = np.sum((predict_label == y_batch.cpu()).numpy())
                val_loss += batch_loss.item()
                val_acc += acc


        print('\n====> Epoch: {} \nTrain:\nCE_Loss: {:.6f} | Accuracy: {:.6f} \nValidation:\nCE_loss: {:.6f} | Accuracy: {:.6f}'.format(
            epoch+1, train_loss/train_cnt, train_acc/len(x_train), val_loss/val_cnt, val_acc/len(x_val)))

        Train_loss_list.append(train_loss/train_cnt)
        Train_acc_list.append(train_acc/len(x_train))
        Val_loss_list.append(val_loss/val_cnt)
        Val_acc_list.append(val_acc/len(x_val))
        
        #Checkpoint
        if (val_acc > best_accuracy):
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'RNN_weights/RNN_'+str(val_acc/len(x_val))+'.pkt')
            print ('Saving Improved Model(val_acc = %.6f)...' % (val_acc/len(x_val)))

        print('-'*88)
    
    with open('./checkpoint/Q2/train_loss.pkl', 'wb') as fp:
        pickle.dump(Train_loss_list, fp)
    with open('./checkpoint/Q2/train_acc.pkl', 'wb') as fp:
        pickle.dump(Train_acc_list, fp)
    with open('./checkpoint/Q2/val_loss.pkl', 'wb') as fp:
        pickle.dump(Val_loss_list, fp)
    with open('./checkpoint/Q2/val_acc.pkl', 'wb') as fp:
        pickle.dump(Val_acc_list, fp)
    
    
def main():
    x_train = torch.load('data/x_train_vgg16.pth')
    y_train = torch.load('data/y_train_vgg16.pth')
    x_val = torch.load('data/x_valid_vgg16.pth')
    y_val = torch.load('data/y_valid_vgg16.pth')

    y_train = np.array(y_train)
    y_val = np.array(y_val)

    print("load data done!!!")

    train(50, x_train, y_train, x_val, y_val)

    
if __name__ == '__main__':
    main()
