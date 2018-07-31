import numpy as np
import os
import torch
import torchvision 
import skvideo.io
import skimage.transform
from utils import to_var
from reader import readFullLengthVideos
from reader import readFullLengthLabels
import pickle

#filepath = 'HW5_data/TrimmedVideos/video/train'
#tag_path = 'HW5_data/TrimmedVideos/label/gt_train.csv'

def get_data(video_path, tag_path, model):
    if torch.cuda.is_available():
        model.cuda()

    feature_size = 512*7*7
    x_all = []
    video_categories = sorted([file for file in os.listdir(video_path) if file.startswith('OP')])
    
    print(len(video_categories))

    for video in video_categories:
        print("Loading frames from video: {} ...".format(video))
        filepath = os.path.join(video_path, video)
        frames = readFullLengthVideos(filepath)

        x_feat = []
        with torch.no_grad():
            for i in range(0, len(frames)):
                x_input = frames[i].unsqueeze(0).cuda()
                features = model(x_input).detach().cpu().numpy().reshape(-1,512*7*7)
                x_feat.append(features)
                
        x_all.append(torch.from_numpy(np.vstack(x_feat)))
        print ("x_feat.shape: ", x_all[-1].shape)
    
    
    y_all = readFullLengthLabels(tag_path)
    y_all = [torch.LongTensor(labels) for labels in y_all]
    
    print('reading video from {}... finished'.format(video_path))
    
    return x_all, y_all
    

def main():
    train_videopath = 'HW5_data/FullLengthVideos/videos/train'
    train_tagpath = 'HW5_data/FullLengthVideos/labels/train'
    valid_videopath = 'HW5_data/FullLengthVideos/videos/valid'
    valid_tagpath = 'HW5_data/FullLengthVideos/labels/valid'

    print('loading VGG16 model...')
    model = torchvision.models.vgg16(pretrained=True).features
    
    x_train, y_train = get_data(train_videopath, train_tagpath, model)
    x_valid, y_valid = get_data(valid_videopath, valid_tagpath, model)
    
    torch.save(x_train, 'data/x_train_full.pth')
    torch.save(y_train, 'data/y_train_full.pth')
    torch.save(x_valid, 'data/x_valid_full.pth')
    torch.save(y_valid, 'data/y_valid_full.pth')
    
    
if __name__ == '__main__':
    main()
