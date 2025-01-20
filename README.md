# LLM4Beginner
This is a simple repository for LLM by a student in AI, with the purpose of enhancing LLM engineering ability. I am also a beginner in LLM, so please feel free to discuss if you find any problems. This repository aims to build Large (Vision) Language Models from scratch in a unified format with `transformers` models, so it is not highly dependent on this package and useful for understanding the basic knowledge of LLM.

## This repository contains
1. A simple fully connected model for generation.
2. A simple Decoder-Only Transformer for generation. A checkpoint can be found in [link](https://drive.google.com/file/d/1QCl3M-9j1RtiAbu8LE_p8rOWb2v2ttko/view?usp=sharing)
3. Attention map visualization support for transformer-based models.
4. A Vision Transformer (VIT) for image classification.

## Install
The codes are bulit with PyTorch, and tokenizer from huggingface is used. So please install the following packages.

    pip install torch transformers einops

You should create three folders: data & ckpts for data storage and checkpoints storage, and runs for TensorBoard log.

    mkdir data ckpts runs

The data folder contains files in text for training, e.g. an e-book in txt file.
**Important:** If you do not have local cache for tokenizer, please uncomment `os.environ['TRANSFORMERS_OFFLINE'] = '1'`

## Train
If you want to train the Decoder-Only Transformer, run the following command

    python trainDecoder.py

As for the VIT, run

    python trainVIT.py

This will automatically download CIFAR-10 dataset for training. You can change it to any dataset easily.

Please note that you should first modify `config.py` according to your device.

## Generate
A generation task is given in `Generate.ipynb`, the generation includes random generation (start with random tokens) and context generation (start with given context). Before generating, please make sure that your trained weights is ready. Enjoy!

## Attention Visualization
An example for attention map visualization is in `AttentionMap.ipynb`. This note book first execute context generation task and then plot the attention score for specific layer and heads. Currently this notebook only supports the visualization of text transformer, and support of VIT will be included in the future.
