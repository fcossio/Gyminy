curl "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o miniconda.#!/bin/sh
sudo apt-get update
bash miniconda.sh
sudo apt-get install
export PATH=~/miniconda3/bin:$PATH
exit

sudo apt-get install gcc

conda env create -f environment.yml

conda activate rbs
pip install tensorflow
pip install -r requirements.txt
