# Bridging the Bosphorus: Advancing Turkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking

<p align="center">
  <img width=65% src="figures/bridge_image.png" alt="Bridging Bosphorus logo" title="Bridging-Bosphorus-logo">
</p>

- **Project Website:** [Bridging-the-Bosphorus](https://emrecanacikgoz.github.io/Bridging-the-Bosphorus/)
- **Paper:** *[Bridging the Bosphorus: Advancing Turkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking](https://arxiv.org/abs/2405.04685)*

This repository contains resources related to the development and evaluation of Large Language Models (LLMs) for the Turkish language, focusing on addressing the challenges faced by low-resource languages. The work presented here explores various strategies for training and evaluating LLMs in underrepresented languages, with a special emphasis on Turkish.

## Overview

Large Language Models (LLMs) are increasingly important in various fields, highlighting the need for high-quality models in underrepresented languages. This study investigates the unique challenges faced by low-resource languages, including data scarcity, model selection, evaluation, and computational limitations. The study includes an in-depth analysis to assess the impact of training strategies, model choices, and data availability on the performance of LLMs designed for underrepresented languages.

## Usage

For researchers and developers interested in LLMs for under-resourced languages, this repository serves as a comprehensive guide for building and evaluating LLMs in resource-constrained environments. The resources provided here aim to support future research in Turkish language processing and the broader field of Natural Language Processing (NLP) for under-resourced languages.

## Requirements

Before you start, please install the necessary packages here: [requirements.txt](./requirements.txt)

## Pretraining from Scratch

The scripts and the bash files used for pretraining the Hamza LLM Series: consisting of Hamza-small (124M), Hamza-medium (354M), Hamza-large (772M), and Hamza-xlarge (1.3b) models can be found under the directory `pretraining/`, with a separate bash file for each model.

The dataset to train the model can be prepared using the scripts under `pretraining/data/culturax/`.

## Fine-tuning
For fine-tune a Llama model on generated alpaca-tr data, please refer to `finetuning` ([here](./finetuning/readme.md)) directory and follow readme.

## Evaluations

For detailed information about Evaluations on TruthfulQA-TR and ARC-TR Turkish question-answering dataset, please read the documentation here [Evaluations](./evaluation/readme.md)

## Citation

If you use this software or our paper, please cite them:

<pre>
@inproceedings{acikgoz-etal-2024-bridging,
    title = "Bridging the Bosphorus: Advancing {T}urkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking",
    author = "Acikgoz, Emre Can  and
      Erdogan, Mete  and
      Yuret, Deniz",
    editor = {S{\"a}lev{\"a}, Jonne  and
      Owodunni, Abraham},
    booktitle = "Proceedings of the Fourth Workshop on Multilingual Representation Learning (MRL 2024)",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.mrl-1.21/",
    doi = "10.18653/v1/2024.mrl-1.21",
    pages = "242--268",
    abstract = "Large Language Models (LLMs) are becoming crucial across various fields, emphasizing the urgency for high-quality models in underrepresented languages. This study explores the unique challenges faced by low-resource languages, such as data scarcity, model selection, evaluation, and computational limitations, with a special focus on Turkish. We conduct an in-depth analysis to evaluate the impact of training strategies, model choices, and data availability on the performance of LLMs designed for underrepresented languages. Our approach includes two methodologies: (i) adapting existing LLMs originally pretrained in English to understand Turkish, and (ii) developing a model from the ground up using Turkish pretraining data, both supplemented with supervised fine-tuning on a novel Turkish instruction-tuning dataset aimed at enhancing reasoning capabilities. The relative performance of these methods is evaluated through the creation of a new leaderboard for Turkish LLMs, featuring benchmarks that assess different reasoning and knowledge skills. Furthermore, we conducted experiments on data and model scaling, both during pretraining and fine-tuning, simultaneously emphasizing the capacity for knowledge transfer across languages and addressing the challenges of catastrophic forgetting encountered during fine-tuning on a different language. Our goal is to offer a detailed guide for advancing the LLM framework in low-resource linguistic contexts, thereby making natural language processing (NLP) benefits more globally accessible."
}
</pre>

## Acknowledgements

This work is supported in part provided by the KUIS AI Center. The numerical calculations reported in this paper were fully/partially performed at TUBITAK ULAKBIM, High Performance and Grid Computing Center (TRUBA resources). Last but not least, we also acknowledge VSB – Technical University of Ostrava, IT4Innovations National Supercomputing Center, Czech Republic, for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (grant ID: 90254)
