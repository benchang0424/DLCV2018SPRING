#!/bin/bash
python3 plot.py --train_path $1 --out_path $2 --mode vae
python3 plot.py --train_path $1 --out_path $2 --mode gan
python3 plot.py --train_path $1 --out_path $2 --mode acgan