"""
This script gives you two .bin files, one for training and one for validation.
These files contain the tokenized text data in the form of a numpy array.
The data is tokenized using the GPT-2 Byte Pair Encoding (BPE) tokenizer.
The tokenizer is obtained from the tiktoken module.

The data is loaded from the parquet files in the data-parquet directory.
The validation data is loaded from the trnews-64-test.json file.

Output files: train-parquet.bin, val-parquet.bin
"""

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset('parquet', data_files="data-parquet/*.parquet") # 129,486,207,634 tokens
    dataset = dataset.remove_columns(['timestamp', 'url', 'source'])

    # use tr-news dataset for validation
    test_dataset = load_dataset('json', data_files="trnews-64-test.json") # 6,720,970 tokens

    # remove columns we don't need
    test_dataset = test_dataset.remove_columns(['timestamp', 'url', 'source'])
    dataset['val'] = test_dataset['train']
    split_dataset = dataset
    print(split_dataset)


    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}-parquet-all.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):

            # batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            # write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
