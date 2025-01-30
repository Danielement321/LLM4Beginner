# LLM4Beginner
This is a simple repository for LLM by a student in AI, with the purpose of enhancing LLM engineering ability. I am also a beginner in LLM, so please feel free to discuss if you find any problems. This repository aims to build Large (Vision) Language Models from scratch in a unified format with `transformers` models, so it is not highly dependent on this package and useful for understanding the basic knowledge of LLM.

## This repository contains
1. A simple fully connected model with SwiGELU for generation.
2. A simple Decoder-Only Transformer for generation, which can be seen as the simple FFN model with attention mechanism.
3. Attention map visualization support for transformer-based models.
4. A Vision Transformer (VIT) for image classification.

## Install
The codes are bulit with PyTorch, and tokenizer from huggingface is used. So please install the following packages.

    pip install torch transformers einops

You should create three folders: data & ckpts for data storage and checkpoints storage, and runs for TensorBoard log.

    mkdir data ckpts runs

The data folder contains files in text for training, e.g. an e-book in txt file.

**IMPORTANT** If you do not have local cache for tokenizer, please uncomment `os.environ['TRANSFORMERS_OFFLINE'] = '1'`. If you encounter network problem, especially for those in China mainland, please uncomment `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'`.

## Train
### Simple FFN Model For Generation
For the SimpleModel, run

    python SimpleModel/trainSimpleModel.py

### Decoder-Only Transformer For Generation
For the Decoder-Only Transformer, if you want to better understand how does the model accept data, run the following command

    python LLM/trainDecoder.py

If you want to train the Decoder-Only Transformer with `Trainer` from transformer, run

    python LLM/trainWithTrainer.py

### Lightweight VIT For Image Classification
As for the VIT, run

    python VIT/trainVIT.py

This will automatically download CIFAR-10 dataset for training. You can change it to any dataset easily.

### Simple Vision Language Model
The VLM is trained in two stages.
- Pretrain 

    For pretrain, first prepare data and revise the `data_path` in `VLM/PreTrain.py`. We recommand using LLaVA Instruct CC3M Pretrain 595K data for a simple demo. After adjusting the data_path, run

        python VLM/PreTrain.py

    **Note:** The keys of original LLaVA pretrain data is not in line with the Qwen tokenizer, `_convert_keys` method is applied for conversion. If you want to train with custom data, make sure that the data contains "user" and "assistant" keys.

- Visual Instruction Tunning

    We utlize LLaVA 158k data for visual instruction tunning. The data format is same as that described in **Pretrain**. Run

        python VLM/FineTune.py

**IMPORTANT:** You should first modify `config.py` according to your device.

## Generate
A generation task is given in `LLM/Generate.ipynb`, the generation includes random generation (start with random tokens) and context generation (start with given context). Before generating, please make sure that your trained weights is ready. Enjoy!

## Attention Visualization
An example for attention map visualization is in `LLM/AttentionMap.ipynb`. This note book first execute context generation task and then plot the attention score for specific layer and heads. Currently this notebook only supports the visualization of text transformer, and support of VIT will be included in the future.

## Acknowledgement
This project draws inspiration and borrows some code from the following repositories and blogs:
- [wyf3/llm_related](https://github.com/wyf3/llm_related)
- [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- [（徒手搓LLM）逐行代码从0构造一个LLM——LlaMa篇](https://zhuanlan.zhihu.com/p/1674261485)
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)

We acknowledge for their outstanding and fruitful work!