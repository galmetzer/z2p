#!/bin/bash

CUDA=cu101
#CUDA=cu102
#CUDA=cu111

conda create -n z2p python=3.8
conda activate z2p

pip install torch==1.8.1+$CUDA torchvision==0.9.1+$CUDA torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
