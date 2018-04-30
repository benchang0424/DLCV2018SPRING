#!/bin/bash
wget -O model_fcn8s.h5 https://www.dropbox.com/s/91zums7f1h2j1sg/fcn8s_best.h5?dl=1
python3 test.py -m model_fcn8s.h5 -v $1 -p $2