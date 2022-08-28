# CEM: Machine-Human Chatting Handoff via Causal-Enhance Module
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)
![GitHub](https://img.shields.io/badge/Qrange%20-group-orange)

This repository is the implementation of "CEM: Machine-Human Chatting Handoff via Causal-Enhance Module" [[paper]](https://arxiv.org/abs/?) on Clothe and Makeup2 datasets. Our paper has been accepted for presentation at ???. You can also check with the [??? proceeding version](???).


## Introduction

CEM solves the problem of Machine-Human Chatting Handoff, establishing the causal graph of MHCH based on these two variables, which is a simple yet effective module and can be easy to plug into the existing MHCH methods. 

<p align="center">
  <img src="https://github.com/Qrange-group/CEM/blob/master/images/dialog.png" width="300" height="600">
</p>

## Requirement

Activate an enviroment of Python 3.7, then `sh env.sh`.

## Data Format

Our experiments are conducted based on two publicly available Chinese customer service dialogue datasets, namely Clothes and Makeup2, collected by [Song et al. (2019)](https://github.com/songkaisong/ssa) from [Taobao](https://www.taobao.com/). 

- Each pkl file is a data list that includes dialogue samples. The content lists of the dataset can be seen in `data_loader.py`. 

- The vocab.pkl contains a vocabulary class which contains the pre-trained glove word embeddings of token ids.

## Usage

- Train the model (including training, validation, and testing)

```bash
python -u -W ignore main.py --task train --model cem --data makeup2

```

- Test the model

```bash
python -u -W ignore main.py --task test --model cem --data makeup2 --model_path pretrained_model_dir

```

## Citing SPENet

```
```

## Acknowledgments

Many thanks to [LauJames](https://github.com/LauJames) for [his Tensorflow framework](https://github.com/LauJames/RSSN) for MHCH task.
