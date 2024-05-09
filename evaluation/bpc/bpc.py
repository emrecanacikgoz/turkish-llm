#!/usr/bin/env python
# See https://huggingface.co/docs/transformers/perplexity
# Usage: bpc.py -m facebook/xglm-564M -d trnews-64 -n 100000

from tqdm import tqdm
import logging
from logging import info
logging.basicConfig(level=logging.INFO)

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader
log2 = 0.6931471805599453


def load_model(model):
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
    #llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
    llm.eval()                # default=eval. this effects dropout etc but still records in backward graph: https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc
    llm.requires_grad_(False) # default=True. saves memory, does not record in backward graph. better to use with torch.no_grad()?
    # llm.half() # gives inf loss?
    #llm.to("cuda")            # default=cpu.
    return llm


def get_max_length(model):
    if hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    elif hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    else:
        info("Cannot find max_length, guessing 1024")
        max_length = 1024
    return max_length


def get_vocab_size(model):
    if hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    else:
        info("Cannot determine vocab_size")
        vocab_size = 0
    return vocab_size


def load_data(data):
    if data == "trnews-64":
        with open("data/trnews-64.test.raw", "r", encoding="utf-8") as file:
            text = file.read()
    else:
        if data != "wikitext":
            info(f"Only trnews-64 and wikitext are currently supported, loading wikitext instead")
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "".join(test["text"])
    return text


def tokenize(model, text):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokens = []
    for line in tqdm(text.splitlines()): # line-by-line is significantly faster for xglm
        if line:
            line = "\n\n" + line + "\n\n"
            tokens += tokenizer.encode(line)
    tokens = torch.tensor(tokens)
    return tokens


def batch(tokens, max_length, batchsize):
    ntokens = tokens.numel()
    start = 0
    batches = []
    while start < ntokens-1:
        if start + max_length * batchsize < ntokens:
            end = start + max_length * batchsize
            x = tokens[start:end].reshape(batchsize, max_length)
            y = tokens[start+1:end+1].reshape(batchsize, max_length)
        elif start + max_length < ntokens:
            batches = (ntokens - start) // max_length
            end = start + max_length * batches
            x = tokens[start:end].reshape(batches, max_length)
            y = tokens[start+1:end+1].reshape(batches, max_length)
        else:
            length = ntokens-start-1
            end = ntokens-1
            x = tokens[start:end].reshape(1, length)
            y = tokens[start+1:end+1].reshape(1, length)
        batches.append((x,y))
        start = end
    return batches


def nll(llm, batches):
    sum = cnt = 0
    for (x,y) in tqdm(batches):
        with torch.no_grad():
            logits = llm(x.to(llm.device)).logits # [B,T,V]
            logits = logits.view(-1, logits.size(-1)) # [B*T,V]
            targets = y.view(-1).to(llm.device)     # [B*T]
            sum += cross_entropy(logits, targets, reduction="sum")
            cnt += targets.size(0)
    return (sum, cnt)


def causal_lm_eval(model="gpt2", data="wikitext", window=0, nchars=0, batchsize=1):

    info(f"Loading {model}...")
    llm = load_model(model)
    max_length = get_max_length(llm)
    vocab_size = get_vocab_size(llm)
    info(f"{type(llm)} with max_length={max_length}, vocab_size={vocab_size}")

    info(f"Loading {data}...")
    text = load_data(data)
    info(f"Read {len(text)} chars.")
    if nchars <= 0:
        nchars = len(text)
    else:
        info(f"Taking the first {nchars} chars.")
        text = text[0:nchars]

    info(f"Tokenizing...")
    tokens = tokenize(model, text)
    ntokens = tokens.numel()
    info(f"Got {ntokens} tokens from {nchars} chars => {nchars/ntokens} chars per token")

    info("Batching...")
    if window > max_length or window < 1:
        info(f"Using window={max_length}")
        window = max_length
    batches = batch(tokens, window, batchsize)
    info(f"Got {len(batches)} batches from {ntokens} tokens with window={window} batchsize={batchsize}")

    info("Calculating nll")
    (sum_nll, ntokens_predicted) = nll(llm, batches)
    assert ntokens_predicted == ntokens - 1
    nchars_predicted = len(text)
    sum_bits  = sum_nll/log2
    nll_per_token = sum_nll/ntokens_predicted
    nll_per_char  = sum_nll/nchars_predicted
    bits_per_token = sum_bits / ntokens_predicted
    bits_per_char = sum_bits / nchars_predicted
    token_ppl = torch.exp(sum_nll/ntokens_predicted)
    char_ppl  = torch.exp(sum_nll/nchars_predicted)
    info(f"ntok={ntokens_predicted} nchr={nchars} nvoc={vocab_size} sum_nll={sum_nll} sum_bits={sum_bits} nll/tok={nll_per_token} nll/char={nll_per_char} token_ppl={token_ppl} char_ppl={char_ppl} bits/tok={bits_per_token} bits/char={bits_per_char}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Causal LM evaluation")
    parser.add_argument("-m", "--model", default="gpt2", type=str, help="Model id")
    parser.add_argument("-d", "--data", default="wikitext", type=str, help="Data: wikitext or trnews-64")
    parser.add_argument("-w", "--window", default=0, type=int, help="Context length, uses model max_length by default")
    parser.add_argument("-b", "--batchsize", default=1, type=int, help="Batch size")
    parser.add_argument("-n", "--nchars", default=0, type=int, help="Number of chars to test, whole data by default")
    args, unknown = parser.parse_known_args()
    info(args)
    if unknown:
        info(f"Warning: bpc.py: Unrecognized options: {unknown}")
    causal_lm_eval(model=args.model, data=args.data, window=args.window, nchars=args.nchars, batchsize=args.batchsize)
    
# TODO: try 16/8/4 bit to see how bpc changes
# TODO: try smaller window sizes to see how bpc changes
# TODO: figure out how to test encoder-decoder models like flan

# Sample bpc results on trnews-64 first 100K chars:
# facebook/xglm-564M: 1.1479
# facebook/xglm-1.7B: 1.0272
# facebook/xglm-2.9B: 0.9721
# facebook/xglm-4.5B: 0.9835
# facebook/xglm-7.5B: 0.9250

