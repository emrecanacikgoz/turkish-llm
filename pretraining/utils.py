import torch
import math
import numpy as np


def get_batch(split, train_data, val_data, block_size, batch_size, device_type, device):
    """
    Sample a batch of sequences from either the training or validation data.
    
    Args:
        split (str): either 'train' or 'val'
        train_data (np.array): training data
        val_data (np.array): validation data
        block_size (int): length of each sequence
        batch_size (int): number of sequences in a batch
        device_type (str): either 'cuda' or 'cpu'
        device (torch.device): device to move the data to

    Returns:
        x (torch.Tensor): input sequences of shape (batch_size, block_size)
        y (torch.Tensor): target sequences of shape (batch_size, block_size)
    """

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, eval_iters, ctx, train_data, val_data, block_size, batch_size, device_type, device):
    """
    Estimate the loss over either the training or validation data using many batches.

    Args:
        model (nn.Module): model class
        eval_iters (int): number of iterations to estimate the loss
        ctx (torch.autograd.grad_mode): context manager for autograd
        train_data (np.array): training data
        val_data (np.array): validation data
        block_size (int): length of each sequence
        batch_size (int): number of sequences in a batch
        device_type (str): either 'cuda' or 'cpu'
        device (torch.device): device to move the data to

    Returns:
        out (dict): dictionary containing the estimated loss over the training and validation data
    
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(
                split=split, 
                train_data=train_data, 
                val_data=val_data, 
                block_size=block_size, 
                batch_size=batch_size, 
                device_type=device_type, 
                device=device,
            )
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """
    Get the learning rate at a given iteration using a cosine decay scheduler with warmup.

    Args:
        iter_num (int): current iteration number
        warmup_iters (int): number of warmup iterations
        learning_rate (float): initial learning rate
        lr_decay_iters (int): number of iterations to decay the learning rate
        min_lr (float): minimum learning rate
    
    Returns:
        lr (float): learning rate at the given iteration

    """
    # 1) linear warmup for warmup_iters steps
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)