#!/bin/bash
wget -O model_fcn32s.h5 https://www.dropbox.com/s/5md2nv1zizzyxst/fcn32s_best.h5?dl=1
python3 test.py -m model_fcn32s.h5 -v $1 -p $2