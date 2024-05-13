# Bridging the Bosphorus: Advancing Turkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking

<img width=60% src="figures/bridge_image.png" alt="Bridging Bosphorus logo" title="Bridging-Bosphorus-logo">

- **Project Website:** [Bridging-the-Bospohorus](https://emrecanacikgoz.github.io/Bridging-the-Bosphorus/)
- **Paper:** *[Bridging the Bosphorus: Advancing Turkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking](https://arxiv.org/abs/2405.04685)*

This repository contains resources related to the development and evaluation of Large Language Models (LLMs) for the Turkish language, focusing on addressing the challenges faced by low-resource languages. The work presented here explores various strategies for training and evaluating LLMs in underrepresented languages, with a special emphasis on Turkish.

## Overview

Large Language Models (LLMs) are increasingly important in various fields, highlighting the need for high-quality models in underrepresented languages. This study investigates the unique challenges faced by low-resource languages, including data scarcity, model selection, evaluation, and computational limitations. The study includes an in-depth analysis to assess the impact of training strategies, model choices, and data availability on the performance of LLMs designed for underrepresented languages.

## Usage

For researchers and developers interested in LLMs for under-resourced languages, this repository serves as a comprehensive guide for building and evaluating LLMs in resource-constrained environments. The resources provided here aim to support future research in Turkish language processing and the broader field of Natural Language Processing (NLP) for under-resourced languages.

### Requirements

Before you start, please install the necessary packages here: [equirements.txt](./requirements.txt)

### Pretraining from Scratch

The scripts and the bash files used for pretraining the Hamza LLM Series: consisting of Hamza-small (124M), Hamza-medium (354M), Hamza-large (772M), and Hamza-xlarge (1.3b) models can be found under the directory `pretraining/`, with a separate bash file for each model.

The dataset to train the model can be prepared using the scripts under `pretraining/data/culturax/`.

### Pretraining

The scripts and the bash files used for pretraining the Hamza LLM Series: consisting of Hamza-small (124M), Hamza-medium (354M), Hamza-large (772M), and Hamza-xlarge (1.3b) models can be found under the directory "pretraining/", with a separate bash file for each model.

## Evaluations

For detailed information about LLM Bits-Per-Character (BPC) Evaluations, TruthfulQA-TR and ARC-TR Turkish question-answering dataset translation and evaluation, please read the documentation here [Model Deployment](./evaluation/README.md)

## Citation

If you use this software or our paper, please cite them:
<pre>
@misc{acikgoz2024bridging,
            title={Bridging the Bosphorus: Advancing Turkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking}, 
            author={Emre Can Acikgoz and Mete Erdogan and Deniz Yuret},
            year={2024},
            eprint={2405.04685},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
}
</pre>

## Acknowledgements

This work is supported in part provided by the KUIS AI Center. The numerical calculations reported in this paper were fully/partially performed at TUBITAK ULAKBIM, High Performance and Grid Computing Center (TRUBA resources). Last but not least, we also acknowledge VSB – Technical University of Ostrava, IT4Innovations National Supercomputing Center, Czech Republic, for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (grant ID: 90254)
