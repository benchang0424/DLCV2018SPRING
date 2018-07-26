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
from reader import readFullLengthVideos
from reader import readFullLengthLabels


def batch_padding(x_batch): 
    seq_tensors = x_batch.unsqueeze(1)
    lengths = [len(seq_tensors)]
    seq_tensors = to_var(seq_tensors)
    return seq_tensors, lengths


def Get_data(video_path):
    model = torchvision.models.vgg16(pretrained=True).features
    if torch.cuda.is_available():
        model.cuda()

    feature_size = 512*7*7
    x_all = []
    video_categories = sorted([file for file in os.listdir(video_path) if file.startswith('OP')])

    for video in video_categories:
        print("Loading frames from video: {} ...".format(video))
        filepath = os.path.join(video_path, video)
        frames = readFullLengthVideos(filepath)

        x_feat = []
        with torch.no_grad():
            for i in range(0, len(frames)):
                x_input = frames[i].unsqueeze(0).cuda()
                features = model(x_input).detach().cpu().numpy().reshape(-1, 512*7*7)
                x_feat.append(features)
                
        x_all.append(torch.from_numpy(np.vstack(x_feat)))
    
    print('reading video from {}... finished'.format(video_path))   
    return x_all


def PlotLearningCurve(OUT_DIR):
    with open('./checkpoint/Q3/train_loss.pkl', 'rb') as fp:
        train_loss = pickle.load(fp)
    with open('./checkpoint/Q3/train_acc.pkl', 'rb') as fp:
        train_acc = pickle.load(fp)
    with open('./checkpoint/Q3/val_acc.pkl', 'rb') as fp:
        val_acc = pickle.load(fp)
    with open('./checkpoint/Q3/val_loss.pkl', 'rb') as fp:
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

    filename = os.path.join(OUT_DIR, 'fig3_2.jpg')
    plt.savefig(filename)

    print("Plot Learning Curve Done!")


def Predict(x_val, video_dir, OUT_DIR):
    model = Seq_Classifier()
    model.load_state_dict(torch.load('Seq_model.pkt'))
    if torch.cuda.is_available():
        model.cuda()

    result = []
    acc = 0
    with torch.no_grad():
        model.eval()
        for i in range(0, len(x_val)):
            x_batch, batch_lengths = batch_padding(x_val[i])
            output = model(x_batch, batch_lengths)
            predict_label = torch.argmax(output,1).cpu().data
            result.append(predict_label.numpy())

    video_name_list = sorted(os.listdir(video_dir)) 
    
    idx = 0
    for name in video_name_list:
        filename = os.path.join(OUT_DIR, name+'.txt')
        cnt = 0
        print(len(result[idx]))
        with open(filename, 'w') as f:
            for item in result[idx]:
                cnt += 1
                if cnt != len(result[idx]):
                    f.write("%s\n" % item)
                else:
                    f.write("%s" % item)
        idx += 1

    print("Predict Done!")


def main(args):
    OUT_DIR = args.out_path
    video_dir = args.video_dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    x_val= Get_data(video_dir)

    #PlotLearningCurve(OUT_DIR)
    Predict(x_val, video_dir, OUT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW5 Seq2Seq inference')
    parser.add_argument('--video_dir', help='validation video directory', type=str)
    parser.add_argument('--out_path', help='output figure directory', type=str)
    args = parser.parse_args()

    main(args)
