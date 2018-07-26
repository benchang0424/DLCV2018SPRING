import numpy as np
from keras.models import load_model
import skimage.io
from read import *
from PIL import Image
import argparse
import os

BATCH_SIZE = 10


def test(x_test, MODEL_DIR, OUTPUT_DIR):

    print(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    predict_model = load_model(MODEL_DIR)
    pred = predict_model.predict(x_test,batch_size=BATCH_SIZE, verbose=1)
    pred = np.argmax(pred, 3)

    for i in range(pred.shape[0]):
        mask_pred = np.zeros((512, 512, 3), dtype=np.uint8)
        mask  = np.empty((512, 512), dtype=np.uint8)
        mask [ pred[i] == 0] = 3  # (Cyan: 011) Urban land 
        mask [ pred[i] == 1] = 6  # (Yellow: 110) Agriculture land 
        mask [ pred[i] == 2] = 5  # (Purple: 101) Rangeland 
        mask [ pred[i] == 3] = 2  # (Green: 010) Forest land 
        mask [ pred[i] == 4] = 1  # (Blue: 001) Water 
        mask [ pred[i] == 5] = 7  # (White: 111) Barren land 
        mask [ pred[i] == 6] = 0  # (Black: 000) Unknown 
        mask_pred[:,:,0] = (mask//4) % 2
        mask_pred[:,:,1] = (mask//2) % 2
        mask_pred[:,:,2] = mask % 2
        mask_pred = mask_pred * 255
        image = Image.fromarray(mask_pred,'RGB')

        image.save('{}/{:0>4}_mask.png'.format(OUTPUT_DIR,i))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model name', type=str)
    parser.add_argument('-v', '--val', help='validation images directory', type=str)
    parser.add_argument('-p', '--pred', help='output images directory', type=str)
    args = parser.parse_args()

    MODEL_DIR = args.model
    VAL_DIR = args.val
    OUTPUT_DIR = args.pred

    print(OUTPUT_DIR)
    x_test = read_images(VAL_DIR)
    x_test = x_test.astype(np.float32) / 255.0
    test(x_test, MODEL_DIR, OUTPUT_DIR)
