import numpy as np
import torch


def so2_R2_vf(x: torch.Tensor, shift: torch.Tensor = None) -> torch.Tensor:
    '''
    Vector field for rotation in R^2.
    Works for batched input of shape (B, 2).
    If shift is given (shape broadcastable to (B, 2)), rotates around that point.
    '''
    if shift is not None:
        x = x - shift
    vf = torch.zeros_like(x)
    vf[:, 0] = -x[:, 1]
    vf[:, 1] = x[:, 0]
    return vf

def scaling2_R2_vf(x: torch.Tensor, shift: torch.Tensor = None) -> torch.Tensor:
    '''
    Vector field for uniform scaling in R^2.
    Works for batched input of shape (B, 2).
    If shift is given (shape broadcastable to (B, 2)), scales around that point.
    The flow of dx/dt = x - c is x(t) = c + (x0 - c)*exp(t), i.e. radial scaling from c.
    '''
    if shift is not None:
        x = x - shift
    vf = torch.zeros_like(x)
    vf[:, 0] = x[:, 0]
    vf[:, 1] = x[:, 1]
    return vf


