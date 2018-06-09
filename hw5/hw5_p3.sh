#!/bin/bash
wget -O Seq_model.pkt https://www.dropbox.com/s/stjk75le4m9h4wa/Seq_0.5885.pkt?dl=1
python3 seq_inference.py --video_dir $1 --out_path $2
