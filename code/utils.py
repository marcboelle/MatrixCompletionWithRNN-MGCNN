import torch
from typing import Union
import numpy as np

def compute_cheb_polynomials(L : Union[np.array, torch.Tensor], 
                             ord_cheb : int) -> torch.Tensor: 
    """
    Computes Chebyshev polynomials up to the order ord_cheb-1 for a given Laplacian matrix L

    Returns : 

    torch.Tensor, a tensor of Chebyshev polynomials of L
    """
    if isinstance(L, np.ndarray):
        L = torch.Tensor(L)

    list_cheb = torch.zeros(ord_cheb, L.shape[0], L.shape[1])

    for k in range(ord_cheb):
        if (k==0):
            list_cheb[k] = torch.eye(L.shape[0])
        elif (k==1):
            list_cheb[k] = torch.Tensor(L)
        else:
            list_cheb[k] = 2*L @ list_cheb[k-1]  - list_cheb[k-2]

    return list_cheb


def frobenius_norm(tensor : torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(tensor ** 2))