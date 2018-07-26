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


def batch_padding(train_x, train_y):
    seq_tensors = train_x.unsqueeze(1)
    label = torch.LongTensor(np.array(train_y))

    lengths = [len(train_x)]
    seq_tensors = to_var(seq_tensors)
    label = to_var(label)
    return seq_tensors, label, lengths


def Get_data(video_path, tag_path):
    model = torchvision.models.vgg16(pretrained=True).features
    if torch.cuda.is_available():
        model.cuda()
    file_dict = getVideoList(tag_path)
    feature_size = 512*7*7
    x, y = [], []
    print(len(file_dict['Video_index']))

    with torch.no_grad():
        for i in range(len(file_dict['Video_index'])):
            frames = readShortVideo(video_path, file_dict['Video_category'][i],file_dict['Video_name'][i])
            if frames.shape[0] > 120:
                output_1 = model(torch.from_numpy(frames[0:120,:,:,:]).cuda()).detach().cpu().reshape(-1,feature_size)
                output_2 = model(torch.from_numpy(frames[120:,:,:,:]).cuda()).detach().cpu().reshape(-1,feature_size)
                output = torch.cat((output_1, output_2), 0)
            else:
                output = model(torch.from_numpy(frames).cuda()).detach().cpu().reshape(-1,feature_size)
            
            x.append(output)
            y.append(int(file_dict['Action_labels'][i]))
            print('\rreading image from {}...{}'.format(video_path,i),end='')

    print('\rreading image from {}...finished'.format(video_path))
    
    return x,y


def PlotLearningCurve(OUT_DIR):
    with open('./checkpoint/Q2/train_loss.pkl', 'rb') as fp:
        train_loss = pickle.load(fp)
    with open('./checkpoint/Q2/train_acc.pkl', 'rb') as fp:
        train_acc = pickle.load(fp)
    with open('./checkpoint/Q2/val_acc.pkl', 'rb') as fp:
        val_acc = pickle.load(fp)
    with open('./checkpoint/Q2/val_loss.pkl', 'rb') as fp:
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

    filename = os.path.join(OUT_DIR, 'fig2_2.jpg')
    plt.savefig(filename)

    print("Plot Learning Curve Done!")


def Predict(x_val, y_val, OUT_DIR):

    model = RNN_Classifier()
    model.load_state_dict(torch.load('RNN_model.pkt'))
    
    if torch.cuda.is_available():
        model.cuda()

    result, acc = [],[]
    with torch.no_grad():
        model.eval()
        for i in range(0, len(x_val)):
            
            x_batch, y_batch, batch_lengths = batch_padding(x_val[i], y_val[i])

            output = model(x_batch, batch_lengths)

            predict_label = torch.argmax(output,1).cpu().data
            acc.append((predict_label == y_batch.cpu()).numpy())
            result.append(predict_label.numpy()[0])

    print(len(acc))
    acc = np.mean(acc)
    print("validation acc : ", acc)

    filename = os.path.join(OUT_DIR, 'p2_result.txt')
    cnt = 0
    with open(filename, 'w') as f:
        for item in result:
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
    y_val = np.array(y_val)

    #PlotLearningCurve(OUT_DIR)
    Predict(x_val, y_val, OUT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW5 CNN inference')
    parser.add_argument('--video_dir', help='validation video directory', type=str)
    parser.add_argument('--csv_dir', help='ground_truth csv directory', type=str)
    parser.add_argument('--out_path', help='output figure directory', type=str)
    args = parser.parse_args()

    main(args)
