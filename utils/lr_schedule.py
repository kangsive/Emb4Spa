import torch
import torch.nn as nn
import torch.optim as optim
import math

def linear_warmup_then_exp_decay(optimizer, base_lr, warmup_epochs, total_epochs):
    """Create a scheduler with linear warmup and exponential decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float): Base learning rate.
        num_warmup_steps (int): Number of steps to linearly warmup.
        num_total_steps (int): Total number of steps during training.

    Returns:
        LambdaLR: Lambda learning rate scheduler.
    """
    def lr_lambda(epoch):
        min_lr = 1e-6
        if epoch < warmup_epochs:
            lr = base_lr * epoch / warmup_epochs 
        else:
            lr = min_lr + (base_lr - min_lr) * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            # print(lr)
            
        # for param_group in optimizer.param_groups:
        #     if "lr_scale" in param_group:
        #         param_group["lr"] = lr * param_group["lr_scale"]
        #     else:
        #         param_group["lr"] = lr
        return lr
            
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def adjust_learning_rate(optimizer, epoch, base_lr, warmup_epochs, total_epochs):
    min_lr = 1e-6
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (base_lr - min_lr) * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
        return lr