#!/bin/bash
wget -O RNN_model.pkt https://www.dropbox.com/s/ng074l81uim1b87/RNN_0.51837.pkt?dl=1
python3 rnn_inference.py --video_dir $1 --csv_dir $2 --out_path $3