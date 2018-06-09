#!/bin/bash
wget -O CNN_model.pkl https://www.dropbox.com/s/s4mtyminqp0aah7/CNN_model.pkl?dl=1
python3 cnn_inference.py --video_dir $1 --csv_dir $2 --out_path $3
