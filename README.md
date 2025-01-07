# LLM4Beginner
This is a simple repository for LLM by a student in AI, with the purpose of enhancing LLM engineering ability. Since I am also a beginner in LLM, so please feel free to discuss if you find any problems.

## This repository contains
1. A simple fully connected model for generation.
2. A simple Decoder-Only Transformer for generation.

## Install
The codes are bulit with PyTorch, and tokenizer from huggingface is used. So please install the following packages.

    pip install torch transformers

You should create two folders: data & ckpts for data storage and checkpoints storage.

    mkdir data ckpts

The data folder contains files in text for training, e.g. an e-book in txt file.

## Train
If you want to train the Decoder-Only Transformer, run the following command

    python trainDecoder.py

Please note that you should first modify `config.py` according to your device.

## Generate
A random generation task is given in `RandomGenerate.ipynb`. Before generating, please make sure that your trained weights is ready.