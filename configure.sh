#!/usr/bin/env bash

folder=$(pwd)
echo ${folder}

#mkvirtualenv venv -p python3
#workon venv
virtualenv -p python3 ./venv
cd ./venv/bin
source activate
pip install numpy scipy matplotlib scikit-image scikit-learn ipython
#pip install -r ${folder}/requirements.txt
deactivate

sudo apt-get install python-opencv
