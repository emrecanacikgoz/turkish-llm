import numpy as np

data = np.memmap("/truba/home/eacikgoz/hamza/turkish-llm/pretraining/data/culturax/train-parquet-all.bin", dtype=np.uint16, mode='r')
print(data.shape)
data = np.memmap("/truba/home/eacikgoz/hamza/turkish-llm/pretraining/data/culturax/val-parquet-all.bin", dtype=np.uint16, mode='r')
print(data.shape)