import torch
from torch import tensor
from torch.nn import Module

def fit(model:Module, epochs:int, bs:int, n:int, x_train:tensor, y_train:tensor, loss_func, lr:torch.float64):
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()
    
    return model