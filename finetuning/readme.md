# **Fine-tuning**
This part only includes alpaca fine-tuning on Turkish instruction tuning dataset `alpaca-tr` which is generated from scratch using ChatGPT.

### **1️⃣ Environment Setup**
To fine-tune the model on a Turkish instruction dataset, first set up a new environment:

```bash
conda create --name llm_tr_finetuning python=3.12 -y
conda activate llm_tr
```

### **2️⃣ Training Configuration**
Modify the configuration as needed before running fine-tuning. Training will be conducted on single-GPU.

```bash
python finetune.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir path/to/output \
  --data-path alpaca-tr.json \
  --batch_size 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 1\
  --learning_rate 1e-4 \
  --cutoff_len 4096 \
  --val_set_size 0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules '[gate_proj, down_proj, up_proj]' \
  --train_on_inputs False \
  --group_by_length False \
  --prompt_template_name alpaca \
  --lr_scheduler 'cosine' \
  --warmup_steps 100
```

For multi-GPU training:
```bash
torchrun --nproc_per_node=8 finetune.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --output_dir path/to/output \
  --data-path alpaca-tr.json \
  --batch_size 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 1\
  --learning_rate 1e-4 \
  --cutoff_len 4096 \
  --val_set_size 0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules '[gate_proj, down_proj, up_proj]' \
  --train_on_inputs False \
  --group_by_length False \
  --prompt_template_name alpaca \
  --lr_scheduler 'cosine' \
  --warmup_steps 100
```
or you can simply run `finetune.sh`.

## Merge LoRA Adapters
Once your training is done, you should merge your LoRA weights with base LLM:

```bash
source merge.sh
```
Don't forget to specify your paths before your merge in `merge.sh` file.

## **📌 Notes**
- Ensure you replace `path/to/model` with the actual model path after fine-tuning.
- Parallelizing evaluation across available GPUs can speed up processing.
- The environment versions (`python=3.12` for training) are chosen based on compatibility with respective libraries.

## Citation
If you use this software or our paper, please cite:
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
